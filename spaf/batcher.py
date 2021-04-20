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

from spaf.network_wrap import (
        TModel_wrap, Networks_wrap)
from spaf.data_access import (
        DataAccess, DataAccess_Train, DataAccess_Eval,
        TDataset_over_DataAccess, Train_Meta, Eval_Meta,
        Train_Item, Eval_Item)
from spaf.utils import (
        tfm_video_resize_threaded, tfm_video_random_crop,
        tfm_video_center_crop, tfm_maybe_flip,
        threaded_ocv_resize_clip,
        _get_centerbox, _get_randombox,
        tfm_uncrop_box, tfm_unresize_box,
        TimersWrap)
from spaf.data.video import (OCV_rstats, video_capture_open, video_sample)

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


def _tqdm_str(pbar, ninc=0):
    if pbar is None:
        tqdm_str = ''
    else:
        tqdm_str = 'TQDM[' + pbar.format_meter(
                pbar.n + ninc, pbar.total,
                pbar._time()-pbar.start_t) + ']'
    return tqdm_str


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
        self.optimizer = optimizer
        self.train_routine = train_routine
        self._save_period = save_period

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
        return start_i

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
        start_i = self._restore_model()
        batches_of_vids_left = self.batches_of_vids[start_i+1:]
        pbar = enumerate(batches_of_vids_left, start=start_i+1)
        pbar = tqdm(pbar)
        for i_meta, batch_of_vids in pbar:
            self.train_routine(i_meta, batch_of_vids)
            # Save check
            SAVE = vst.check_step(i_meta, self._save_period)
            SAVE |= (i_meta+1 == self._total)
            if SAVE:
                self._save_model(i_meta)
                self._purge_intermediate_files()


class Isaver_midepoch_eval(vst.isave.Isaver_base):
    def __init__(
            self, folder, batches_of_vids, eval_routine,
            save_period='::'):
        super().__init__(folder, len(batches_of_vids))
        self.batches_of_vids = batches_of_vids
        self.eval_routine = eval_routine
        self._save_period = save_period
        self.result = {}

    def run(self):
        start_i = self._restore()
        batches_of_vids_left = self.batches_of_vids[start_i+1:]
        pbar = enumerate(batches_of_vids_left, start=start_i+1)
        pbar = tqdm(pbar)
        for i_meta, batch_of_vids in pbar:
            video_outputs = self.eval_routine(i_meta, batch_of_vids)
            self.result.update(video_outputs)
            # Save check
            SAVE = vst.check_step(i_meta, self._save_period)
            SAVE |= (i_meta+1 == self._total)
            if SAVE:
                self._save(i_meta)
                self._purge_intermediate_files()
        return self.result


class Batcher_train_basic(object):
    nswrap: Networks_wrap

    def __init__(
            self, nswrap, data_access, batch_size, num_workers):
        self.nswrap = nswrap
        self.data_access = data_access
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ewrap = TrainEpochWrap()

    def execute_epoch(self, batches_of_vids, folder, epoch) -> None:
        isaver = Isaver_midepoch_train(folder, batches_of_vids,
                self.nswrap.model, self.nswrap.optimizer,
                self.train_on_vids, '0::1')
        isaver.run()

    def train_on_vids(self, i_meta, batch_of_vids) -> None:
        time_period = '::10'
        twrap = TimersWrap(['data', 'gpu'])
        twrap.tic('data')

        # _get_sequences_batch_train_dataloader_v2
        tdataset = TDataset_over_DataAccess(self.data_access, batch_of_vids)
        loader = torch.utils.data.DataLoader(
            tdataset, batch_size=self.batch_size,
            collate_fn=self.data_access.collate_batch,
            shuffle=False, drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True)

        for i, train_item in enumerate(loader):
            twrap.toc('data'); twrap.tic('gpu')
            output_ups, loss, target_ups = \
                self.nswrap.forward_model_for_training(
                    train_item['X'], train_item['X_plus'], train_item['target'])
            self.ewrap.update(loss, output_ups, target_ups)
            loss.backward()
            self.nswrap.optimizer.step()
            self.nswrap.optimizer.zero_grad()
            twrap.toc('gpu'); twrap.tic('data')
            if vst.check_step(i, time_period):
                log.debug('Execute time {}/{} - {time}'.format(
                    i, len(loader), time=twrap.time_str))


class Batcher_eval_basic(object):
    nswrap: Networks_wrap

    def __init__(
            self, nswrap, data_access, batch_size, num_workers):
        self.nswrap = nswrap
        self.data_access = data_access
        self.batch_size = batch_size
        self.num_workers = num_workers

    def execute_epoch(self, batches_of_vids, folder, epoch):
        isaver = Isaver_midepoch_eval(folder, batches_of_vids,
                self.eval_on_vids, '0::1')
        output_items = isaver.run()
        return output_items

    def eval_on_vids(self, i_meta, batch_of_vids) -> Dict:
        time_period = '::10'
        twrap = TimersWrap(['data', 'gpu_prepare', 'gpu'])
        twrap.tic('data')
        output_items = {}

        # _get_dict_1_eval_dataloader_v2
        tdataset = TDataset_over_DataAccess(self.data_access, batch_of_vids)
        loader = torch.utils.data.DataLoader(
            tdataset, batch_size=1,
            collate_fn=self.data_access.collate_batch,
            shuffle=False, num_workers=self.num_workers, pin_memory=True)

        for i, eval_item in enumerate(loader):
            twrap.toc('data'); twrap.tic('gpu')
            output_item = self.nswrap.forward_model_for_eval_cpu(
                    eval_item['X'], eval_item['X_plus'],
                    eval_item['stacked_targets'], self.batch_size)
            vid = eval_item['meta'].vid
            output_items[vid] = output_item

            twrap.toc('gpu'); twrap.tic('data')
            if vst.check_step(i, time_period):
                log.debug('Execute time {}/{} - {time}'.format(
                    i, len(loader), time=twrap.time_str))
        return output_items


class BoxDictTrainDataset(torch.utils.data.Dataset):
    def __init__(self, boxdicts):
        super().__init__()
        self.input_size = 224
        self.boxdicts = boxdicts
        self.initial_resize = 256
        self.norm_mean = np.array(
                [0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std = np.array(
                [0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.boxdicts)

    @staticmethod
    def collate_batch(batch):
        assert len(batch) == 1
        batch0 = batch[0]
        assert isinstance(batch0, collections.abc.Mapping)
        return batch0

    def __getitem__(self, index):
        boxdict = self.boxdicts[index]
        stacked_X_np = []
        stacked_X_plus_np = []
        meta: Train_Meta
        for i, (box, meta) in enumerate(
                zip(boxdict['boxes'], boxdict['train_meta'])):
            with video_capture_open(meta.video_path, np.inf) as vcap:
                frames = np.array(video_sample(vcap, meta.real_sampled_inds))
            frames_rgb = np.flip(frames, -1)

            # // Repeat training video preparation
            X = threaded_ocv_resize_clip(
                    frames_rgb, self.initial_resize)
            p = meta.params['rcrop']
            X = X[:,
                  p['i']:p['i']+p['th'],
                  p['j']:p['j']+p['tw'], :]
            if meta.params['flip']['perform']:
                X = np.flip(X, axis=2).copy()

            if meta.params['flip']['perform']:
                l_, t_, r_, d_ = box
                box_ = np.r_[l_, self.input_size-d_, r_, self.input_size-t_]
            else:
                box_ = box.copy()

            # Cut box and prepare it
            ltrd_1 = tfm_uncrop_box(box_, meta.params['rcrop'])
            ltrd_0 = tfm_unresize_box(ltrd_1, meta.params['resize'])
            l_, t_, r_, d_ = ltrd_0
            X_plus = frames_rgb[:, l_:r_, t_:d_, :]
            if meta.params['flip']['perform']:
                X_plus = np.flip(X_plus, axis=2).copy()
            X_plus = threaded_ocv_resize_clip(X_plus,
                    (self.input_size, self.input_size))
            stacked_X_np.append(X)
            stacked_X_plus_np.append(X_plus)
        stacked_X_np = np.stack(stacked_X_np)
        stacked_X_plus_np = np.stack(stacked_X_plus_np)

        assert stacked_X_np.dtype is np.dtype('uint8'), 'must be uin8'
        assert stacked_X_plus_np.dtype is np.dtype('uint8'), 'must be uin8'

        train_item = Train_Item(
                X=torch.from_numpy(stacked_X_np),
                X_plus=torch.from_numpy(stacked_X_plus_np),
                target=boxdict['train_target'],
                meta=boxdict['train_meta'])
        return train_item


class BoxDictEvalDataset(torch.utils.data.Dataset):
    def __init__(self, boxdicts):
        super().__init__()
        self.input_size = 224
        self.boxdicts = boxdicts
        self.initial_resize = 256
        self.norm_mean = np.array(
                [0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std = np.array(
                [0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.boxdicts)

    @staticmethod
    def collate_batch(batch):
        assert len(batch) == 1
        batch0 = batch[0]
        assert isinstance(batch0, collections.abc.Mapping)
        return batch0

    def __getitem__(self, index):
        boxdict = self.boxdicts[index]
        meta: Eval_Meta = boxdict['eval_meta']

        with video_capture_open(meta.video_path, np.inf) as vcap:
            unique_frames = np.array(video_sample(
                vcap, meta.unique_real_sampled_inds))
        unique_frames_rgb = np.flip(unique_frames, -1)

        # // Repeat eval video preparation (same centercrop everywhere)
        unique_prepared = threaded_ocv_resize_clip(
            unique_frames_rgb, self.initial_resize)
        p = meta.params['ccrop']
        unique_prepared = unique_prepared[:,
                p['i']:p['i']+p['th'],
                p['j']:p['j']+p['tw'], :]

        # // Cut box and prepare it (different boxes)
        stacked_X_plus_np = []
        for i, (box, rel_frame_inds) in enumerate(
                zip(boxdict['boxes'], meta.rel_frame_inds)):
            # Cut box and prepare it
            ltrd_1 = tfm_uncrop_box(box, meta.params['ccrop'])
            ltrd_0 = tfm_unresize_box(ltrd_1, meta.params['resize'])
            l_, t_, r_, d_ = ltrd_0
            # Sample from original frames
            current_frames_rgb = unique_frames_rgb[rel_frame_inds]
            second64 = current_frames_rgb[:, l_:r_, t_:d_, :]
            second64 = threaded_ocv_resize_clip(second64,
                    (self.input_size, self.input_size))
            stacked_X_plus_np.append(second64)
        stacked_X_plus_np = np.stack(stacked_X_plus_np)  # eval_gap,64,224,224,3

        X = unique_prepared[meta.rel_frame_inds]  # eval_gap,64..
        assert X.dtype is np.dtype('uint8'), 'must be uin8'
        assert stacked_X_plus_np.dtype is np.dtype('uint8'), 'must be uin8'

        eval_item = Eval_Item(
                X=torch.from_numpy(X),
                X_plus=torch.from_numpy(stacked_X_plus_np),
                stacked_targets=boxdict['eval_target'],
                meta=meta)
        return eval_item


class Batcher_train_attentioncrop(object):
    nswrap: Networks_wrap
    data_access: DataAccess_Train

    def __init__(
            self, nswrap, data_access, batch_size, num_workers):
        self.nswrap = nswrap
        self.data_access = data_access
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ewrap = TrainEpochWrap()

    def execute_epoch(self, batches_of_vids, folder, epoch) -> None:
        isaver = Isaver_midepoch_train(folder, batches_of_vids,
                self.nswrap.model, self.nswrap.optimizer,
                self.train_on_vids, '0::1')
        isaver.run()

    def _prepare_boxinputs(
            self, loader, twrap, i_meta, time_period='::10'):
        boxdicts = {}
        twrap.tic('data')
        train_item: Train_Item
        for i, train_item in enumerate(loader):
            assert train_item['X_plus'] is None
            twrap.toc('data'); twrap.tic('gpu')
            gradient, boxes = self.nswrap.get_attention_gradient_v2(
                    train_item['X'], train_item['target'])
            boxdict = {
                    'boxes': boxes,
                    'train_target': train_item['target'],
                    'train_meta': train_item['meta']}
            boxdicts[i] = boxdict
            twrap.toc('gpu'); twrap.tic('data')
            if vst.check_step(i, time_period):
                log.debug('Prepare time {}/{} - {time}'.format(
                    i, len(loader), time=twrap.time_str))
            # if self.debug_enabled:
            #     dfold = vst.mkdir(self._hacky_folder/'debug')
            #     self._debug_boxinputs(
            #             dfold, X_f32c.cpu(), gradient, boxes, i_meta, i)
        return boxdicts

    def train_on_vids(self, i_meta, batch_of_vids) -> None:
        time_period = '::10'
        twrap = TimersWrap(['data', 'gpu'])
        twrap.tic('data')

        # Prepare box dicts
        boxinput_dataset = TDataset_over_DataAccess(
                self.data_access, batch_of_vids)
        boxinput_loader = torch.utils.data.DataLoader(
            boxinput_dataset, batch_size=self.batch_size,
            collate_fn=self.data_access.collate_batch,
            shuffle=False, drop_last=True,
            num_workers=self.num_workers, pin_memory=True)
        boxdicts = self._prepare_boxinputs(
                boxinput_loader, twrap, i_meta, time_period)

        # Train on box dicts (looks just like normal training)
        boxdict_dataset = BoxDictTrainDataset(boxdicts)
        boxdict_loader = torch.utils.data.DataLoader(
            boxdict_dataset, batch_size=1,
            collate_fn=boxdict_dataset.collate_batch,
            shuffle=False, num_workers=self.num_workers, pin_memory=True)
        twrap.tic('data')
        for i, train_item in enumerate(boxdict_loader):
            twrap.toc('data'); twrap.tic('gpu')
            output_ups, loss, target_ups = \
                self.nswrap.forward_model_for_training(
                    train_item['X'], train_item['X_plus'], train_item['target'])
            self.ewrap.update(loss, output_ups, target_ups)
            loss.backward()
            self.nswrap.optimizer.step()
            self.nswrap.optimizer.zero_grad()
            twrap.toc('gpu'); twrap.tic('data')
            if vst.check_step(i, time_period):
                log.debug('Execute time {}/{} - {time}'.format(
                    i, len(boxdict_loader), time=twrap.time_str))

            # if self.debug_enabled:
            #     dfold = small.mkdir(self._hacky_folder/'debug')
            #     half = X_f32c.shape[1]//2
            #     input_denorm_ups_bgr = np.flip(upscale_u8(
            #             np_denormalize(X_f32c.cpu().numpy())), axis=-1)
            #     bigger = input_denorm_ups_bgr[:, :half]
            #     smaller = input_denorm_ups_bgr[:, half:]
            #     concat = np.concatenate((bigger, smaller), axis=3)
            #     for j, v in enumerate(concat):
            #         quick_video_save(
            #             dfold/'vis_M{:03d}_I{:03d}_J{:03d}_execute'.format(
            #                 i_meta, i, j), v)

class Batcher_eval_attentioncrop(object):
    nswrap: Networks_wrap
    data_access: DataAccess_Eval

    def __init__(
            self, nswrap, data_access, batch_size, num_workers):
        self.nswrap = nswrap
        self.data_access = data_access
        self.batch_size = batch_size
        self.num_workers = num_workers

    def execute_epoch(self, batches_of_vids, folder, epoch):
        isaver = Isaver_midepoch_eval(folder, batches_of_vids,
                self.eval_on_vids, '0::1')
        output_items = isaver.run()
        return output_items

    def _prepare_boxinputs(self, loader, twrap, i_meta, time_period='::10'):
        boxdicts = {}
        twrap.tic('data')
        eval_item: Eval_Item
        for i, eval_item in enumerate(loader):
            assert eval_item['X_plus'] is None
            twrap.toc('data'); twrap.tic('gpu')
            gradient, boxes = self.nswrap.get_attention_gradient_v2(
                    eval_item['X'], eval_item['stacked_targets'])
            boxdict = {
                    'boxes': boxes,
                    'eval_target': eval_item['stacked_targets'],
                    'eval_meta': eval_item['meta']}
            boxdicts[i] = boxdict
            twrap.toc('gpu'); twrap.tic('data')
            if vst.check_step(i, time_period):
                log.debug('Prepare time {}/{} - {time}'.format(
                    i, len(loader), time=twrap.time_str))
        return boxdicts

    def eval_on_vids(self, i_meta, batch_of_vids) -> Dict:
        time_period = '::10'
        twrap = TimersWrap(['data', 'gpu_prepare', 'gpu'])
        twrap.tic('data')
        output_items = {}

        # Prepare box dicts
        boxinput_dataset = TDataset_over_DataAccess(
                self.data_access, batch_of_vids)
        boxinput_loader = torch.utils.data.DataLoader(
            boxinput_dataset, batch_size=1,
            collate_fn=self.data_access.collate_batch,
            shuffle=False, num_workers=self.num_workers, pin_memory=True)
        boxdicts = self._prepare_boxinputs(
                boxinput_loader, twrap, i_meta, time_period)

        # Eval on box dicts (looks just like normal eval)
        boxdict_dataset = BoxDictEvalDataset(boxdicts)
        boxdict_loader = torch.utils.data.DataLoader(
            boxdict_dataset, batch_size=1,
            collate_fn=boxdict_dataset.collate_batch,
            shuffle=False, num_workers=self.num_workers, pin_memory=True)
        for i, eval_item in enumerate(boxdict_loader):
            twrap.toc('data'); twrap.tic('gpu')
            output_item = self.nswrap.forward_model_for_eval_cpu(
                    eval_item['X'], eval_item['X_plus'],
                    eval_item['stacked_targets'], self.batch_size)
            vid = eval_item['meta'].vid
            output_items[vid] = output_item

            twrap.toc('gpu'); twrap.tic('data')
            if vst.check_step(i, time_period):
                log.debug('Execute time {}/{} - {time}'.format(
                    i, len(boxdict_loader), time=twrap.time_str))
        return output_items
