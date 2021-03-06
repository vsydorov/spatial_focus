import logging
import collections
from typing import (Dict, List, TypedDict)
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import default_collate

import vst

from spaf.network_wrap import (Networks_wrap)
from spaf.data_access import (
        DataAccess, DataAccess_Train, DataAccess_Eval,
        TDataset_over_DataAccess, Train_Meta, Eval_Meta,
        TF_params_grouped, Train_Item, Train_Item_collated, Eval_Item,
        reapply_grouped_transforms, unapply_grouped_transforms_and_cut_box)
from spaf.utils import (
        tfm_video_resize_threaded, tfm_video_random_crop,
        tfm_video_center_crop, tfm_maybe_flip,
        TF_params_resize, TF_params_crop, TF_params_flip,
        _get_centerbox, _get_randombox,
        tfm_uncrop_box, tfm_unresize_box,
        TimersWrap)
from spaf.data.video import (
        OCV_rstats, video_capture_open, video_sample,
        video_writer_open, suffix_helper)

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


# Dataloader logic


def collate_train_items(batch: List[Train_Item]) -> Train_Item_collated:
    X = default_collate([x['X'] for x in batch])
    if batch[0]['X_plus'] is not None:
        X_plus = default_collate([x['X_plus'] for x in batch])
    else:
        X_plus = None
    target = default_collate([x['target'] for x in batch])
    train_meta = [x['train_meta'] for x in batch]
    collated: Train_Item_collated = {
            'X': X, 'X_plus': X_plus,
            'target': target, 'train_meta': train_meta}
    return collated


# def fake_collate_batch0(batch):
#     assert len(batch) == 1
#     batch0 = batch[0]
#     assert isinstance(batch0, collections.abc.Mapping)
#     return batch0


def fake_collate_batch0(batch):
    assert len(batch) == 1
    batch0 = batch[0]
    assert isinstance(batch0, collections.abc.Mapping)
    result = {}
    for k, v in batch0.items():
        if isinstance(v, torch.Tensor) and \
                (torch.utils.data.get_worker_info() is not None):
            v.storage().share_memory_()
        result[k] = v
    return result


def get_train_dataloader(
        tdataset, batch_size, num_workers):
    dloader = torch.utils.data.DataLoader(
        tdataset, batch_size=batch_size,
        collate_fn=collate_train_items,
        shuffle=False, drop_last=True,
        num_workers=num_workers, pin_memory=True)
    return dloader

def get_batch0_dataloader(tdataset, num_workers):
    dloader = torch.utils.data.DataLoader(
        tdataset, batch_size=1,
        collate_fn=fake_collate_batch0, shuffle=False,
        num_workers=num_workers, pin_memory=False)
    return dloader


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
        pbar = tqdm(pbar, total=len(batches_of_vids_left))
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
        pbar = tqdm(pbar, total=len(batches_of_vids_left))
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

    def train_on_vids(self, i_meta, batch_of_vids) -> None:
        twrap = TimersWrap(['data', 'gpu'])

        tdataset = TDataset_over_DataAccess(self.data_access, batch_of_vids)
        dloader = get_train_dataloader(tdataset, self.batch_size, self.num_workers)
        train_item: Train_Item_collated
        twrap.tic('data')
        for i, train_item in enumerate(dloader):
            twrap.toc('data'); twrap.tic('gpu')
            output_ups, loss, target_ups = \
                self.nswrap.forward_model_for_training(
                    train_item['X'], train_item['X_plus'], train_item['target'])
            self.ewrap.update(loss, output_ups, target_ups)
            loss.backward()
            self.nswrap.optimizer.step()
            self.nswrap.optimizer.zero_grad()
            twrap.toc('gpu'); twrap.tic('data')

        if vst.check_step(i_meta, '::1'):
            log.debug('Execute time at i_meta {} - {}'.format(i_meta, twrap.time_str))


class Batcher_eval_basic(object):
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

    def eval_on_vids(self, i_meta, batch_of_vids) -> Dict:
        output_items = {}
        twrap = TimersWrap(['data', 'gpu'])

        tdataset = TDataset_over_DataAccess(self.data_access, batch_of_vids)
        dloader = get_batch0_dataloader(tdataset, self.num_workers)
        eval_item: Eval_Item
        twrap.tic('data')
        for i, eval_item in enumerate(dloader):
            twrap.toc('data'); twrap.tic('gpu')
            output_item = self.nswrap.forward_model_for_eval_cpu(
                    eval_item['X'], eval_item['X_plus'],
                    eval_item['stacked_targets'], self.batch_size)
            vid = eval_item['eval_meta'].vid
            output_items[vid] = output_item
            twrap.toc('gpu'); twrap.tic('data')

        if vst.check_step(i_meta, '::1'):
            log.debug('Execute time at i_meta {} - {}'.format(i_meta, twrap.time_str))
        return output_items


class Boxdict_train(TypedDict):
    boxes: np.ndarray
    train_target: torch.Tensor
    train_meta: List[Train_Meta]


class Boxdict_eval(TypedDict):
    boxes: np.ndarray
    eval_target: torch.Tensor
    eval_meta: Eval_Meta


class BoxDictTrainDataset(torch.utils.data.Dataset):
    boxdicts: Dict[int, Boxdict_train]

    def __init__(self, boxdicts, input_size):
        super().__init__()
        self.boxdicts = boxdicts
        self.input_size = input_size

    def __len__(self):
        return len(self.boxdicts)

    def __getitem__(self, index) -> Train_Item_collated:
        boxdict = self.boxdicts[index]
        stacked_X_np_ = []
        stacked_X_plus_np_ = []
        train_meta: Train_Meta
        for i, (box, train_meta) in enumerate(
                zip(boxdict['boxes'], boxdict['train_meta'])):
            assert train_meta.params is not None
            with video_capture_open(train_meta.video_path, np.inf) as vcap:
                frames = np.array(video_sample(vcap, train_meta.real_sampled_inds))
            frames_rgb = np.flip(frames, -1)
            # Repeat preparations
            X = reapply_grouped_transforms(frames_rgb, train_meta.params)
            # Cut box, prepare
            X_plus = unapply_grouped_transforms_and_cut_box(
                    box, frames_rgb, train_meta.params, self.input_size)
            stacked_X_np_.append(X)
            stacked_X_plus_np_.append(X_plus)
        stacked_X_np = np.stack(stacked_X_np_)
        stacked_X_plus_np = np.stack(stacked_X_plus_np_)

        train_item = Train_Item_collated(
                X=torch.from_numpy(stacked_X_np),
                X_plus=torch.from_numpy(stacked_X_plus_np),
                target=boxdict['train_target'],
                train_meta=boxdict['train_meta'])
        return train_item


class BoxDictEvalDataset(torch.utils.data.Dataset):
    boxdicts: Dict[int, Boxdict_eval]

    def __init__(self, boxdicts, input_size):
        super().__init__()
        self.boxdicts = boxdicts
        self.input_size = input_size

    def __len__(self):
        return len(self.boxdicts)

    def __getitem__(self, index) -> Eval_Item:
        boxdict = self.boxdicts[index]
        eval_meta: Eval_Meta = boxdict['eval_meta']
        assert eval_meta.params is not None
        assert eval_meta.rel_frame_inds is not None

        with video_capture_open(eval_meta.video_path, np.inf) as vcap:
            unique_frames = np.array(video_sample(
                vcap, eval_meta.unique_real_sampled_inds))
        unique_frames_rgb = np.flip(unique_frames, -1)

        # // Repeat eval video preparation (same centercrop everywhere)
        unique_prepared = reapply_grouped_transforms(
                unique_frames_rgb, eval_meta.params)
        X = unique_prepared[eval_meta.rel_frame_inds]  # eval_gap,64..

        # // Cut box and prepare it (different boxes)
        stacked_X_plus_np_ = []
        for i, (box, rel_frame_inds) in enumerate(
                zip(boxdict['boxes'], eval_meta.rel_frame_inds)):
            current_frames_rgb = unique_frames_rgb[rel_frame_inds]
            X_plus = unapply_grouped_transforms_and_cut_box(
                    box, current_frames_rgb, eval_meta.params, self.input_size)
            stacked_X_plus_np_.append(X_plus)
        stacked_X_plus_np = np.stack(stacked_X_plus_np_)  # eval_gap,64,224,224,3

        eval_item = Eval_Item(
                X=torch.from_numpy(X),
                X_plus=torch.from_numpy(stacked_X_plus_np),
                stacked_targets=boxdict['eval_target'],
                eval_meta=eval_meta)
        return eval_item


def scale_01(X):
    max_ = np.max(X)
    min_ = np.min(X)
    spread = max_-min_
    X = (X-min_)/spread
    return X


def upscale_u8(X):
    return (X*255).astype(np.uint8)


def cv_box(img, box_,
        box_color=(30, 30, 255),
        box_thickness: int = 2):
    # Box configs
    box_pt1 = tuple(box_[:2])
    box_pt2 = tuple(box_[2:])
    cv2.rectangle(img, box_pt1, box_pt2, box_color, box_thickness)
    return img


def quick_video_save(filepath, X_BGR):
    assert len(X_BGR.shape) == 4
    H, W = X_BGR.shape[1:3]
    video_path = suffix_helper(filepath, 'XVID')
    with video_writer_open(video_path, (W, H), 10, 'XVID') as vout:
        for frame in X_BGR:
            vout.write(frame)


def _debug_stacked(debug_folder, X, X_plus, i_meta, i):
    concat = np.concatenate((X.numpy(), X_plus.numpy()), axis=3)
    concat_bgr = np.flip(concat, axis=-1)
    for j, v in enumerate(concat_bgr):
        quick_video_save(debug_folder/
            'vis_M{:03d}_I{:03d}_J{:03d}_execute'.format(i_meta, i, j), v)


class Batcher_train_attentioncrop(object):
    nswrap: Networks_wrap
    data_access: DataAccess_Train

    def __init__(
            self, nswrap, data_access, batch_size, num_workers, debug_enabled):
        self.nswrap = nswrap
        self.data_access = data_access
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_enabled = debug_enabled

        self.ewrap = TrainEpochWrap()

    def execute_epoch(self, batches_of_vids, folder, epoch) -> None:
        self._execute_folder = folder
        isaver = Isaver_midepoch_train(folder, batches_of_vids,
                self.nswrap.model, self.nswrap.optimizer,
                self.train_on_vids, '0::1')
        isaver.run()

    def _debug_boxinputs(self, gradient, boxes, train_item, i_meta, i):
        debug_folder = vst.mkdir(self._execute_folder/'debug')
        gradient_np = gradient.cpu().numpy()  # B, 1, H, W, 1
        Xs = [scale_01(g)[0, :, :, 0] for g in gradient_np]  # [H, W]
        heatmaps = [cv2.applyColorMap(
            upscale_u8(x), cv2.COLORMAP_JET) for x in Xs]
        # Make frames
        X_bgr = np.flip(train_item['X'].numpy(), axis=-1).copy()
        for j, (input_bgr, heatmap) in enumerate(zip(X_bgr, heatmaps)):
            vis_frames_ = []
            box = boxes[j]
            for f in input_bgr:
                f = cv_box(f.copy(), box[[1, 0, 3, 2]])
                f = cv2.addWeighted(f, 1, heatmap, 0.5, 0)
                vis_frames_.append(f)
            vis_frames = np.array(vis_frames_)
            quick_video_save(
                debug_folder/'vis_M{:03d}_I{:03d}_J{:03d}_prepare'.format(
                    i_meta, i, j), vis_frames)

    def train_on_vids(self, i_meta, batch_of_vids) -> None:
        twrap = TimersWrap(['data', 'gpu'])

        # Prepare box dicts
        tdataset = TDataset_over_DataAccess(self.data_access, batch_of_vids)
        dloader = get_train_dataloader(tdataset, self.batch_size, self.num_workers)
        boxdicts: Dict[int, Boxdict_train] = {}
        train_item: Train_Item_collated
        twrap.tic('data')
        for i, train_item in enumerate(dloader):
            assert train_item['X_plus'] is None
            twrap.toc('data'); twrap.tic('gpu')
            gradient, boxes = self.nswrap.get_attention_gradient_v2(
                    train_item['X'], train_item['target'])
            boxdict: Boxdict_train = {'boxes': boxes,
                    'train_target': train_item['target'],
                    'train_meta': train_item['train_meta']}
            boxdicts[i] = boxdict
            twrap.toc('gpu'); twrap.tic('data')
            if self.debug_enabled:
                self._debug_boxinputs(gradient, boxes, train_item, i_meta, i)

        # Train on box dicts (looks just like normal training)
        tdataset_bd = BoxDictTrainDataset(boxdicts, self.data_access.input_size)
        dloader_bd = get_batch0_dataloader(tdataset_bd, self.num_workers)
        twrap.tic('data')
        for i, train_item in enumerate(dloader_bd):
            assert train_item['X_plus'] is not None
            twrap.toc('data'); twrap.tic('gpu')
            output_ups, loss, target_ups = \
                self.nswrap.forward_model_for_training(
                    train_item['X'], train_item['X_plus'], train_item['target'])
            self.ewrap.update(loss, output_ups, target_ups)
            loss.backward()
            self.nswrap.optimizer.step()
            self.nswrap.optimizer.zero_grad()
            twrap.toc('gpu'); twrap.tic('data')
            if self.debug_enabled:
                _debug_stacked(vst.mkdir(self._execute_folder/'debug'),
                        train_item['X'], train_item['X_plus'], i_meta, i)

        if vst.check_step(i_meta, '::1'):
            log.debug('Execute time at i_meta {} - {}'.format(i_meta, twrap.time_str))

class Batcher_eval_attentioncrop(object):
    nswrap: Networks_wrap
    data_access: DataAccess_Eval

    def __init__(
            self, nswrap, data_access, batch_size, num_workers, debug_enabled):
        self.nswrap = nswrap
        self.data_access = data_access
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_enabled = debug_enabled

    def execute_epoch(self, batches_of_vids, folder, epoch):
        isaver = Isaver_midepoch_eval(folder, batches_of_vids,
                self.eval_on_vids, '0::1')
        output_items = isaver.run()
        return output_items

    def eval_on_vids(self, i_meta, batch_of_vids) -> Dict:
        output_items = {}
        twrap = TimersWrap(['data', 'gpu'])

        # Prepare box dicts
        tdataset = TDataset_over_DataAccess(self.data_access, batch_of_vids)
        dloader = get_batch0_dataloader(tdataset, self.num_workers)
        boxdicts: Dict[int, Boxdict_eval] = {}
        eval_item: Eval_Item
        twrap.tic('data')
        for i, eval_item in enumerate(dloader):
            assert eval_item['X_plus'] is None
            twrap.toc('data'); twrap.tic('gpu')
            gradient, boxes = self.nswrap.get_attention_gradient_v2(
                    eval_item['X'], eval_item['stacked_targets'])
            boxdict: Boxdict_eval = {'boxes': boxes,
                    'eval_target': eval_item['stacked_targets'],
                    'eval_meta': eval_item['eval_meta']}
            boxdicts[i] = boxdict
            twrap.toc('gpu'); twrap.tic('data')

        # Eval on box dicts (looks just like normal eval)
        tdataset_bd = BoxDictEvalDataset(boxdicts, self.data_access.input_size)
        dloader_bd = get_batch0_dataloader(tdataset_bd, self.num_workers)
        for i, eval_item in enumerate(dloader_bd):
            twrap.toc('data'); twrap.tic('gpu')
            output_item = self.nswrap.forward_model_for_eval_cpu(
                    eval_item['X'], eval_item['X_plus'],
                    eval_item['stacked_targets'], self.batch_size)
            vid = eval_item['eval_meta'].vid
            output_items[vid] = output_item
            twrap.toc('gpu'); twrap.tic('data')

        if vst.check_step(i_meta, '::1'):
            log.debug('Execute time at i_meta {} - {}'.format(i_meta, twrap.time_str))
        return output_items


def boxwise_extract_upscale(X, boxes, input_size: int):
    # Conversion to avoid crash at interpolate
    X_f32_unnorm = X.type(torch.FloatTensor)  # type: ignore
    # Extract box, rescale to (input_size x input_size)
    eboxes224_ = []
    for i, box in enumerate(boxes):
        l_, t_, r_, d_ = box
        ebox = X_f32_unnorm[i, :, l_:r_, t_:d_, :]
        ebox_prm = ebox.permute(0, 3, 1, 2)
        ebox224_prm = torch.nn.functional.interpolate(
                ebox_prm, (input_size, input_size),
                mode='bilinear', align_corners=True)
        ebox224 = ebox224_prm.permute(0, 2, 3, 1)
        eboxes224_.append(ebox224)
    X_plus = torch.stack(eboxes224_)
    X_plus = X_plus.type(torch.uint8)
    return X_plus


class Batcher_train_rescaled_attentioncrop(object):
    nswrap: Networks_wrap
    data_access: DataAccess_Train

    def __init__(
            self, nswrap, data_access, batch_size, num_workers, debug_enabled):
        self.nswrap = nswrap
        self.data_access = data_access
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_enabled = debug_enabled

        self.ewrap = TrainEpochWrap()

    def execute_epoch(self, batches_of_vids, folder, epoch) -> None:
        self._execute_folder = folder
        isaver = Isaver_midepoch_train(folder, batches_of_vids,
                self.nswrap.model, self.nswrap.optimizer,
                self.train_on_vids, '0::1')
        isaver.run()

    def train_on_vids(self, i_meta, batch_of_vids) -> None:
        twrap = TimersWrap(['data', 'gpu'])

        tdataset = TDataset_over_DataAccess(self.data_access, batch_of_vids)
        dloader = get_train_dataloader(tdataset, self.batch_size, self.num_workers)
        train_item: Train_Item_collated
        twrap.tic('data')
        for i, train_item in enumerate(dloader):
            twrap.toc('data'); twrap.tic('gpu')
            gradient, boxes = self.nswrap.get_attention_gradient_v2(
                    train_item['X'], train_item['target'])
            X_plus = boxwise_extract_upscale(train_item['X'], boxes,
                     self.data_access.input_size)
            output_ups, loss, target_ups = \
                self.nswrap.forward_model_for_training(
                    train_item['X'], X_plus, train_item['target'])
            self.ewrap.update(loss, output_ups, target_ups)
            loss.backward()
            self.nswrap.optimizer.step()
            self.nswrap.optimizer.zero_grad()
            twrap.toc('gpu'); twrap.tic('data')
            if self.debug_enabled:
                _debug_stacked(vst.mkdir(self._execute_folder/'debug'),
                        train_item['X'], X_plus, i_meta, i)

        if vst.check_step(i_meta, '::1'):
            log.debug('Execute time at i_meta {} - {}'.format(i_meta, twrap.time_str))


class Batcher_eval_rescaled_attentioncrop(object):
    nswrap: Networks_wrap
    data_access: DataAccess_Eval

    def __init__(
            self, nswrap, data_access, batch_size, num_workers, debug_enabled):
        self.nswrap = nswrap
        self.data_access = data_access
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_enabled = debug_enabled

    def execute_epoch(self, batches_of_vids, folder, epoch):
        self._execute_folder = folder
        isaver = Isaver_midepoch_eval(folder, batches_of_vids,
                self.eval_on_vids, '0::1')
        output_items = isaver.run()
        return output_items

    def eval_on_vids(self, i_meta, batch_of_vids) -> Dict:
        output_items = {}
        twrap = TimersWrap(['data', 'gpu'])

        tdataset = TDataset_over_DataAccess(self.data_access, batch_of_vids)
        dloader = get_batch0_dataloader(tdataset, self.num_workers)
        eval_item: Eval_Item
        twrap.tic('data')
        for i, eval_item in enumerate(dloader):
            twrap.toc('data'); twrap.tic('gpu')
            gradient, boxes = self.nswrap.get_attention_gradient_v2(
                    eval_item['X'], eval_item['stacked_targets'])
            X_plus = boxwise_extract_upscale(eval_item['X'], boxes,
                     self.data_access.input_size)
            output_item = self.nswrap.forward_model_for_eval_cpu(
                    eval_item['X'], X_plus,
                    eval_item['stacked_targets'], self.batch_size)
            vid = eval_item['eval_meta'].vid
            output_items[vid] = output_item
            twrap.toc('gpu'); twrap.tic('data')
            if self.debug_enabled:
                _debug_stacked(vst.mkdir(self._execute_folder/'debug'),
                        eval_item['X'], X_plus, i_meta, i)

        if vst.check_step(i_meta, '::1'):
            log.debug('Execute time at i_meta {} - {}'.format(i_meta, twrap.time_str))
        return output_items
