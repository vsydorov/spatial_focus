import logging
import re
import collections
from abc import abstractmethod, ABC
from pathlib import Path
from typing import (Dict, List)

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import default_collate

import vst

from spaf.network_wrap import (NWrap)
from spaf.data_access import (
        DataAccess, DataAccess_Train, DataAccess_Eval)
from spaf.trainer import (
        MKinds_Charades, Metrics_Charades, mkinds_to_string)
from spaf.utils import (TimersWrap)

log = logging.getLogger(__name__)


def multilabel_batch_accuracy_topk(outputs, targets, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = targets.size(0)
    maxk = max(topk)
    _, pred = outputs.topk(maxk, 1,
            largest=True, sorted=True)
    pred = pred.t()

    correct = torch.zeros(*pred.shape)
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            correct[i, j] = targets[j, pred[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.div_(batch_size).item())
    return res


class TrainEpochWrap(object):
    def __init__(self):
        self.names = ['loss', 'acc1', 'acc5']
        self.meters = \
                {k: vst.Averager() for k in self.names}

    def update(self, loss, scores, score_target):
        self.meters['loss'].update(loss.item())
        acc1, acc5 = multilabel_batch_accuracy_topk(
                scores, score_target, (1, 5))
        self.meters['acc1'].update(acc1)
        self.meters['acc5'].update(acc5)

    @property
    def meters_str(self):
        meters_str = ' '.join(['{}: {m.last:.4f}({m.avg:.4f})'.format(
                k, m=self.meters[k]) for k in self.names])
        return meters_str


def np_multilabel_batch_accuracy_topk(outputs, targets, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = targets.shape[0]
    maxk = max(topk)
    topk_ids = outputs.argsort(axis=1)[:, ::-1][:, :maxk]

    correct = np.zeros_like(topk_ids)
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            correct[i, j] = targets[i, topk_ids[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(correct_k/batch_size)
    return res


def _map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return _map(fix, gt_array)


def _quick_scores_from_video_outputs(video_outputs) -> Metrics_Charades:
    if len(video_outputs) == 0:
        log.warn('Trying to compute scores on empty video outputs')
        return Metrics_Charades()

    scores_, score_targets_ = zip(*[
            (x['scores'], x['score_target'])
            for x in video_outputs.values()])
    scores_ = np.array([x for x in scores_])
    score_targets_ = np.array([x for x in score_targets_])
    vscores_ = scores_.max(1)
    vscore_targets_ = score_targets_.max(1)
    mAP, wAP, ap = charades_map(vscores_, vscore_targets_)
    acc1, acc5 = np_multilabel_batch_accuracy_topk(
            vscores_,
            vscore_targets_, topk=(1, 5))
    loss = np.mean([x['loss']
        for x in video_outputs.values()])
    train_loss = np.mean([x['train_loss']
        for x in video_outputs.values()])
    return Metrics_Charades(mAP, acc1, acc5, loss, train_loss)


def _tqdm_str(pbar, ninc=0):
    if pbar is None:
        tqdm_str = ''
    else:
        tqdm_str = 'TQDM[' + pbar.format_meter(
                pbar.n + ninc, pbar.total,
                pbar._time()-pbar.start_t) + ']'
    return tqdm_str


class MetaBatcher(ABC):
    meta_twrap: TimersWrap
    sub_twraps: Dict[str, TimersWrap]

    def __init__(self, norm_mean, norm_std,
            size, data_access, nwrap, video_batch_size,
            num_workers, debug_enabled):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.size = size
        self.data_access = data_access
        self.nwrap = nwrap
        self.video_batch_size = video_batch_size
        self.num_workers = num_workers
        self.debug_enabled = debug_enabled

        self.METABATCHNAME = f'metabatch_S{self.size}.pkl'

        self.save_period = '0::1'
        self.log_period = '0::1'
        self.shuffle_vids = True

    def to_gpu_and_normalize(self, X):
        X_f32c = X.type(
                torch.cuda.FloatTensor, non_blocking=True)
        X_f32c /= 255
        X_f32c = (X_f32c-self.norm_mean)/self.norm_std
        return X_f32c

    def _subwrap_avg_times(self):
        times_sum = {}
        times_count = {}
        for k, timers_wrap in self.sub_twraps.items():
            for name, meter in timers_wrap.meters.items():
                times_sum[name] = times_sum.setdefault(name, 0.0) + meter.avg
                times_count[name] = times_count.setdefault(name, 0) + 1
        return times_sum, times_count

    def _get_time_str(self):
        time_str = 'AVG+['
        times_sum, times_count = self._subwrap_avg_times()
        time_str += ' '.join(['{} {:.2f}s({})'.format(k, v, times_count[k])
            for k, v in times_sum.items()])
        time_str +='] {}'.format(self.meta_twrap.time_str)
        return time_str

    @staticmethod
    def _get_intermediate_files(folder, imt_re_finished):
        intermediate_files = {}
        for filename in folder.iterdir():
            matched = re.match(imt_re_finished, filename.name)
            if matched:
                i_meta = int(matched.groupdict()['i_meta'])
                intermediate_files[i_meta] = filename.with_suffix('')

        return intermediate_files

    @staticmethod
    def _purge_intermediate(folder, imt_re_finished):
        # Remove old saved states
        intermediate_files = \
                MetaBatcher._get_intermediate_files(folder, imt_re_finished)
        inds_to_purge = np.sort(np.fromiter(
            intermediate_files.keys(), np.int))[:-2]
        for ind in inds_to_purge:
            filename = intermediate_files[ind]
            (filename.parent/(filename.name + '.finished')).unlink()
            filename.unlink()
        log.debug('Purged {} files'.format(len(inds_to_purge)))

    def _get_vids_metabatches(self, folder, vids, shuffle=True):
        # Shuffle vids, divide into batches
        metabatch_file = folder/self.METABATCHNAME
        if metabatch_file.exists():
            vids_metabatches = vst.load_pkl(metabatch_file)
        else:
            if shuffle is True:
                vids_ = np.random.permutation(vids)
                log.debug('Vids shuffled to {}'.format(vids_))
            else:
                vids_ = vids
            vids_metabatches = vst.leqn_split(vids_, self.size)
            vst.save_pkl(metabatch_file, vids_metabatches)
        return vids_metabatches

    @abstractmethod
    def _save(self, folder, i_meta, N_total):
        raise NotImplementedError()

    @abstractmethod
    def _log(self, epoch, pbar, i_meta, N_total):
        raise NotImplementedError()

    @abstractmethod
    def _process_metabatch(self, i_meta, vids):
        raise NotImplementedError()

    def run(self, vids, folder, epoch):
        vids_metabatches = self._get_vids_metabatches(
                folder, vids, self.shuffle_vids)
        N_total = len(vids_metabatches)

        intermediate_files = self._get_intermediate_files(
                folder, self.imt_re_finished)
        start_i_meta, intermediate_file = max(intermediate_files.items(),
                default=(-1, None))
        self._maybe_restore_intermediate(
                start_i_meta, intermediate_file)
        vids_metabatches_ = vids_metabatches[start_i_meta+1:]

        self._hacky_folder = folder

        pbar = tqdm(vids_metabatches_)
        for i_meta, metabatch in enumerate(pbar, start=start_i_meta+1):
            self._process_metabatch(i_meta, metabatch)
            if vst.check_step(i_meta, self.save_period):
                self._save(folder, i_meta, N_total)
                self._purge_intermediate(folder, self.imt_re_finished)
            if vst.check_step(i_meta, self.log_period):
                self._log(epoch, pbar, i_meta, N_total)
            if i_meta+1 == N_total:
                self._save(folder, i_meta, N_total)
                self._purge_intermediate(folder, self.imt_re_finished)
        return self._results


class TrainMetaBatcher(MetaBatcher):
    def __init__(self, norm_mean, norm_std, size, data_access, nwrap,
            video_batch_size, num_workers, debug_enabled):
        super().__init__(
                norm_mean, norm_std, size, data_access,
                nwrap, video_batch_size, num_workers, debug_enabled)
        self.imt_re_finished = \
                r'mbatch_(?P<i_meta>\d{3}).pth.tar.finished'
        self.ewrap = TrainEpochWrap()

    def _maybe_restore_intermediate(
            self, start_i_meta, intermediate_file):
        if intermediate_file is not None:
            log.info('Loading intermediate training model from {}'.format(
                intermediate_file))
            states = torch.load(intermediate_file)
            self.nwrap.load_state_dicts(states['state_dicts'])
            assert states['i_meta'] == start_i_meta

    @property
    def _results(self):
        return self.ewrap.meters

    def _save(self, folder, i_meta, N_total):
        # Save stuff
        imt_format = 'mbatch_{:03d}.pth.tar'
        savepath = folder/imt_format.format(i_meta)
        state_dicts = self.nwrap.get_state_dicts()
        states = {
            'i_meta': i_meta,
            'state_dicts': state_dicts}
        torch.save(states, str(savepath))
        (savepath.parent/(savepath.name + '.finished')).touch()
        log.debug(('Saved imt trained model i_meta {} path {}'.format(
                    i_meta, savepath)))

    def _log(self, epoch, pbar, i_meta, N_total):
        # Logging
        epoch_str = 'Train Epoch [{}, {}/{}]'.format(
                epoch, i_meta, N_total)
        time_str = self._get_time_str()
        log_str = '{epoch} {meters}\n\t{time} {tqdm}'.format(
                epoch=epoch_str, meters=self.ewrap.meters_str,
                time=time_str, tqdm=_tqdm_str(pbar, 1))
        log.info(log_str)


class EvalMetaBatcher(MetaBatcher):
    def __init__(self, norm_mean, norm_std, size, data_access, nwrap,
            video_batch_size, num_workers, debug_enabled):
        super().__init__(
                norm_mean, norm_std, size, data_access,
                nwrap, video_batch_size, num_workers, debug_enabled)
        self.imt_re_finished = (r'mbatch_(?P<i_meta>\d{3})_'
                r'of_(?P<n_total>\d{3}).pkl.finished')
        self.shuffle_vids = False

    def run(self, vids, folder, epoch, suffix):
        self.suffix = suffix
        return super().run(vids, folder, epoch)

    def _maybe_restore_intermediate(self, start_i_meta, intermediate_file):
        if intermediate_file is None:
            self.video_outputs = {}
        else:
            log.info('Loading intermediate eval video_outputs from {}'.format(
                intermediate_file))
            self.video_outputs = vst.load_pkl(intermediate_file)

    @staticmethod
    def results_from_video_outputs(video_outputs) -> MKinds_Charades:
        metrics_charades = \
                _quick_scores_from_video_outputs(video_outputs)
        metrics_kinds_charades = {'normal': metrics_charades}
        return metrics_kinds_charades

    @property
    def _results(self):
        return self.results_from_video_outputs(self.video_outputs)

    def _save(self, folder, i_meta, N_total):
        # Save stuff
        imt_format = 'mbatch_{:03d}_of_{:03d}.pkl'
        savepath = folder/imt_format.format(i_meta, N_total)
        vst.save_pkl(savepath, self.video_outputs)
        (savepath.parent/(savepath.name + '.finished')).touch()
        log.debug(('Saved intermediate video_outputs imeta {} path {}'.format(
                    i_meta, savepath)))

    def _log(self, epoch, pbar, i_meta, N_total):
        epoch_str = '{} Evaluation Epoch [{}, {}/{}]'.format(
                self.suffix, epoch, i_meta, N_total)
        time_str = self._get_time_str()
        partial_results: MKinds_Charades = \
                self.results_from_video_outputs(self.video_outputs)
        log_str = '{epoch} \n\t{time} {tqdm}\n\tMetrics[{metrics}]'.format(
                epoch=epoch_str,
                metrics=mkinds_to_string(partial_results, ' || '),
                time=time_str,
                tqdm=_tqdm_str(pbar, 1))
        log.info(log_str)

    @staticmethod
    def _criterions_to_cpudict(nwrap, output, target_c):
        output_ups, loss, target_ups = nwrap.eval_criterion(
                output, target_c)
        _, train_loss, _ = nwrap.train_criterion(
                output, target_c)

        output_ups = output_ups.cpu().numpy()
        loss = loss.cpu().item()
        train_loss = train_loss.cpu().item()
        target_ups = target_ups.cpu().numpy()

        return {'scores': output_ups,
                'loss': loss,
                'train_loss': train_loss,
                'score_target': target_ups}


class TrainMetaBatcher_Normal(TrainMetaBatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_twrap = TimersWrap(['execute'])
        self.sub_twraps = {
            'execute': TimersWrap(['data', 'gpu_prepare', 'gpu'])}

    def _execute(
            self, nwrap: NWrap,
            loader, twrap, ewrap, time_period='::10'):
        twrap.tic('data')
        # train_item: Train_Item
        for i, train_item in enumerate(loader):
            train_input_u8, train_target, train_meta = train_item
            twrap.toc('data')
            twrap.tic('gpu_prepare')
            X_f32c = self.to_gpu_and_normalize(train_input_u8)
            train_target_ = train_target.cuda()
            twrap.toc('gpu_prepare')
            twrap.tic('gpu')
            output = nwrap.forward_model(X_f32c)
            output_ups, loss, target_ups = nwrap.train_criterion(
                    output, train_target_)
            ewrap.update(loss, output_ups, target_ups)
            loss.backward()
            nwrap.optimizer_step()
            twrap.toc('gpu'); twrap.tic('data')
            if vst.check_step(i, time_period):
                log.debug('Execute time {}/{} - {time}'.format(
                    i, len(loader), time=twrap.time_str))

    def _process_metabatch(self, i_meta, vids):
        # Train
        self.meta_twrap.tic('execute')
        dataset = TDataset_over_DataAccess(self.data_access, vids)
        loader = _get_sequences_batch_train_dataloader_v2(
                dataset, self.video_batch_size, self.num_workers)
        self._execute(
                self.nwrap, loader,
                self.sub_twraps['execute'], self.ewrap)
        self.meta_twrap.toc('execute')


class EvalMetaBatcher_Normal(EvalMetaBatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_twrap = TimersWrap(['execute'])
        self.sub_twraps = {
            'execute': TimersWrap([
                'data', 'gpu_prepare', 'gpu'])}

    def _execute(
            self, nwrap: NWrap,
            loader, twrap, time_period='::10'):
        twrap.tic('data')
        video_outputs = {}
        # eval_item: Eval_Item
        for i, eval_item in enumerate(loader):
            twrap.toc('data')
            twrap.tic('gpu_prepare')
            X = eval_item['X']

            target_c = eval_item['stacked_targets'].cuda()
            twrap.toc('gpu_prepare')
            twrap.tic('gpu')
            output = nwrap.forward_by_eval_batch_size_efficient(
                X, self.norm_mean, self.norm_std)

            vid = eval_item['meta'].vid
            video_outputs[vid] = self._criterions_to_cpudict(
                    nwrap, output, target_c)

            twrap.toc('gpu'); twrap.tic('data')
            if vst.check_step(i, time_period):
                log.debug('Execute time {}/{} - {time}'.format(
                    i, len(loader), time=twrap.time_str))
        return video_outputs

    def _process_metabatch(self, i_meta, vids):
        # Eval
        self.meta_twrap.tic('execute')
        dataset = TDataset_over_DataAccess(self.data_access, vids)
        loader = _get_dict_1_eval_dataloader_v2(dataset,
                self.num_workers)
        curr_video_outputs = self._execute(
                self.nwrap, loader, self.sub_twraps['execute'])
        self.video_outputs.update(curr_video_outputs)
        self.meta_twrap.toc('execute')


# New batcher philosophy

# Isave classes that manage mid-epoch restoring
class Isaver_midepoch_train(vst.isave.Isaver_base0):
    def __init__(
            self, folder, batches_of_vids,
            model, optimizer, train_routine,
            save_period='::'):
        super().__init__(folder, len(batches_of_vids))
        self.batches_of_vids = batches_of_vids
        self.model = model
        self.train_routine = train_routine
        self.optimizer = optimizer

    def _get_filenames(self, i) -> Dict[str, Path]:
        filenames = super()._get_filenames(i)
        filenames['state_dicts'] = filenames['finished'].with_suffix('.pth.tar')
        return filenames

    def _restore_model(self):
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        start_i, ifiles = max(intermediate_files.items(),
                default=(-1, None))
        if ifiles is not None:
            restore_from = ifiles['state_dicts']
            state_dicts = torch.load(restore_from)
            self.model.load_state_dict(state_dicts['model'])
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            log.debug('Restored model, optimizer from {}'.format(restore_from))

    def _save_model(self, i):
        ifiles = self._get_filenames(i)
        save_to = ifiles['state_dicts']
        vst.mkdir(save_to.parent)
        state_dicts = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}
        torch.save(state_dicts, str(save_to))
        ifiles['finished'].touch()

    def run(self):
        start_i = self._restore()
        batches_of_vids_left = self.batches_of_vids[start_i+1:]
        pbar = enumerate(batches_of_vids_left, start=start_i)
        pbar = tqdm(pbar)
        for i_meta, batch_of_vids in pbar:
            self.train_routine(batch_of_vids)
            # Save check
            SAVE = vst.check_step(i_meta, self._save_period)
            SAVE |= (i_meta+1 == self._total)
            if SAVE:
                self._save(i_meta)
                self._purge_intermediate_files()


class Batcher_train_basic(object):
    def __init__(self, data_access):
        self.data_access = data_access

    def execute_epoch(batches_of_vids, folder, epoch):
        pass
