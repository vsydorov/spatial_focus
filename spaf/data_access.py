"""
DataAccess is a data wrapper that serves as a base for TorchDataset
"""
import numpy as np
import logging
import collections
from pathlib import Path
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import (Any, Optional, Tuple, Union, List, TypedDict)

import torch
from torch.utils.data.dataloader import default_collate

from spaf.utils import (
        tfm_video_resize_threaded, tfm_video_random_crop,
        tfm_video_center_crop, tfm_maybe_flip,
        threaded_ocv_resize_clip,
        _get_centerbox, _get_randombox,
        tfm_uncrop_box, tfm_unresize_box,
        TimersWrap)

from spaf.data.dataset import (
        Sampler_charades, Video_charades)

log = logging.getLogger(__name__)


class DataAccess(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_item(self, vid: str):
        raise NotImplementedError()


class Train_Transform_Params(TypedDict):
    resize: Any
    rcrop: Any
    flip: Any


class Eval_Transform_Params(TypedDict):
    resize: Any
    ccrop: Any


@dataclass
class Train_Meta:
    vid: str
    video_path: Path
    real_sampled_inds: np.ndarray
    do_not_collate: bool = True
    plus_box: Optional[np.ndarray] = None
    params: Optional[Train_Transform_Params] = None


@dataclass
class Eval_Meta:
    vid: str
    video_path: Path
    shifts: np.ndarray
    unique_real_sampled_inds: np.ndarray
    plus_box: Optional[np.ndarray] = None
    rel_frame_inds: Optional[np.ndarray] = None
    params: Optional[Eval_Transform_Params] = None
    tw: Optional[TimersWrap] = None


# Train_Item = Tuple[torch.Tensor, torch.Tensor, Train_Meta]
# Train_Item_plus = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Train_Meta]


class Train_Item(TypedDict):
    X: torch.Tensor
    X_plus: Optional[torch.Tensor]
    target: torch.Tensor
    meta: Train_Meta


class Eval_Item(TypedDict):
    X: torch.Tensor
    X_plus: Optional[torch.Tensor]
    stacked_targets: torch.Tensor
    meta: Eval_Meta


# new_target: str
# target = self._adjust_train_target(X, target)
# def _adjust_train_target(self, X, target):
#     # Target adjustment
#     assert len(X) == len(target)
#     if self.new_target == 'single':
#         pass
#     elif self.new_target == 'append':
#         target = target.repeat(2, 1)
#     else:
#         raise NotImplementedError()
#     return target
#
# def _adjust_eval_target(self, input_, stacked_targets):
#     assert input_.shape[1] == stacked_targets.shape[1]
#     if self.new_target == 'single':
#         pass
#     elif self.new_target == 'append':
#         stacked_targets = stacked_targets.repeat(1, 2, 1)
#     else:
#         raise NotImplementedError()
#     return stacked_targets


def dict_1_collate_v2(batch):
    assert len(batch) == 1
    batch0 = batch[0]
    assert isinstance(batch0, collections.abc.Mapping)
    use_shared_memory = torch.utils.data.dataloader._use_shared_memory
    # log.info('Use shared memory {}'.format(use_shared_memory))
    result = {}
    for k, v in batch0.items():
        if use_shared_memory and isinstance(v, torch.Tensor):
            v.storage().share_memory_()
        result[k] = v
    return result


def _get_dict_1_eval_dataloader_v2(gdataset, num_workers):
    loader = torch.utils.data.DataLoader(
        gdataset, batch_size=1,
        shuffle=False, collate_fn=dict_1_collate_v2,
        num_workers=num_workers, pin_memory=True)
    return loader


def sequence_batch_collate_v2(batch):
    assert isinstance(batch[0], collections.abc.Sequence), \
            'Only sequences supported'
    transposed = zip(*batch)
    collated = []
    for samples in transposed:
        if isinstance(samples[0], collections.abc.Mapping) \
               and 'do_not_collate' in samples[0]:
            c_samples = samples
        elif getattr(samples[0], 'do_not_collate', False) is True:
            c_samples = samples
        else:
            c_samples = default_collate(samples)
        collated.append(c_samples)
    return collated


def _get_sequences_batch_train_dataloader_v2(
        gdataset, batch_size, num_workers, shuffle=True):
    loader = torch.utils.data.DataLoader(
        gdataset, batch_size=batch_size,
        collate_fn=sequence_batch_collate_v2,
        shuffle=shuffle, drop_last=True,
        num_workers=num_workers, pin_memory=True, sampler=None)
    return loader


class TDataset_over_DataAccess(torch.utils.data.Dataset):
    data_access: DataAccess
    vids: List[str]

    def __init__(self, data_access: DataAccess, vids: List[str]):
        super().__init__()
        self.data_access = data_access
        self.vids = vids

    def __getitem__(self, index: int):
        vid = self.vids[index]
        return self.data_access.get_item(vid)

    def __len__(self) -> int:
        return len(self.vids)


class DataAccess_Train(DataAccess):
    sampler: Sampler_charades
    initial_resize: int
    input_size: int
    train_gap: int
    fps: int
    params_to_meta: bool

    def __init__(
            self, dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta, new_target):
        super().__init__()
        self.sampler = Sampler_charades(dataset)
        self.initial_resize = initial_resize
        self.input_size = input_size
        self.train_gap = train_gap
        self.fps = fps
        self.params_to_meta = params_to_meta

    @staticmethod
    def _apply_training_transforms(
            X, initial_resize, input_size
            ) -> Tuple[np.ndarray, Train_Transform_Params]:
        X, resize_params = tfm_video_resize_threaded(X, initial_resize)
        X, rcrop_params = tfm_video_random_crop(
                X, input_size, input_size)
        X, flip_params = tfm_maybe_flip(X)
        assert X.dtype is np.dtype('uint8'), 'must be uin8'
        params: Train_Transform_Params = {
                'resize': resize_params,
                'rcrop': rcrop_params,
                'flip': flip_params}
        return X, params

    def _presample_training_frames(self, vid: str, shift: Optional[float]):
        """
        Retrieve a window of frames at "shift" position
          - if shift not specified - sample random position in the video
        """
        if not self.sampler.labels_present:
            raise RuntimeError('SA does not allow sampling targets. '
                    'Training not supported')

        if shift is None:
            shift = np.random.rand()

        video: Video_charades = self.sampler.get_video(vid)

        real_sampled_inds, sampled_times = \
                self.sampler.sample_frameids_and_times(
                        video, shift, self.train_gap, self.fps)
        frames_u8 = self.sampler.sample_frames(video, real_sampled_inds)
        frames_u8 = np.flip(frames_u8, -1)  # Make RGB
        target = self.sampler.sample_targets(video, sampled_times)

        meta = Train_Meta(
            vid=vid, video_path=video['path'],
            real_sampled_inds=real_sampled_inds)
        return target, meta, frames_u8

    def get_item(
            self, vid: str, shift=None
            ) -> Train_Item:
        meta: Train_Meta
        target, meta, frames_rgb_u8 = \
                self._presample_training_frames(vid, shift)
        X, params = self._apply_training_transforms(frames_rgb_u8,
                self.initial_resize, self.input_size)
        X_tensor = torch.from_numpy(X)
        if self.params_to_meta:
            meta.params = params
        train_item = Train_Item(
                X=X_tensor,
                X_plus=None,
                target=target,
                meta=meta)
        return train_item


class DataAccess_Eval(DataAccess):
    sampler: Sampler_charades
    initial_resize: int
    input_size: int
    train_gap: int
    fps: int
    params_to_meta: bool
    new_target: str
    eval_gap: int

    def __init__(self, dataset,
            initial_resize, input_size,
            train_gap, fps,
            params_to_meta, new_target, eval_gap):
        super().__init__()
        self.sampler = Sampler_charades(dataset)
        self.initial_resize = initial_resize
        self.input_size = input_size
        self.train_gap = train_gap
        self.fps = fps
        self.params_to_meta = params_to_meta
        self.new_target = new_target
        self.eval_gap = eval_gap

    @staticmethod
    def _eval_prepare(X, initial_resize, input_size
                      ) -> Tuple[np.ndarray, Eval_Transform_Params]:
        X, resize_params = tfm_video_resize_threaded(X, initial_resize)
        X, ccrop_params = tfm_video_center_crop(
                X, input_size, input_size)
        assert X.dtype is np.dtype('uint8'), 'must be uin8'
        params: Eval_Transform_Params = {
                'resize': resize_params,
                'ccrop': ccrop_params}
        return X, params

    def _presample_evaluation_frames(self, vid: str):
        """
        Retrieve self.eval_gap windows of frames at regular intervals
        - Since some frames will be repeated multiple times, return unique
          frames only and the index to make non-unique
        """
        shifts = np.linspace(0, 1.0, self.eval_gap)

        video: Video_charades = self.sampler.get_video(vid)

        # Sample frames_inds and times
        sampled2 = [self.sampler.sample_frameids_and_times(
                    video, shift, self.train_gap, self.fps)
                    for shift in shifts]
        all_real_sampled_inds, all_sampled_times = zip(*sampled2)

        unique_real_sampled_inds = \
                np.unique(np.hstack(all_real_sampled_inds))
        all_relative_sampled_inds = []
        for x in all_real_sampled_inds:
            y = np.searchsorted(unique_real_sampled_inds, x)
            all_relative_sampled_inds.append(y)
        stacked_relative_sampled_inds = \
                np.vstack(all_relative_sampled_inds)

        unique_frames = self.sampler.sample_frames(
                video, unique_real_sampled_inds)
        unique_frames_rgb_u8 = np.flip(unique_frames, -1)  # Make RGB

        stacked_targets: Optional[torch.Tensor]
        if self.sampler.labels_present:
            all_targets = [self.sampler.sample_targets(video, sampled_times)
                for sampled_times in all_sampled_times]
            stacked_targets = torch.stack(all_targets)  # N_batch, T, N_class
        else:
            stacked_targets = None

        # Resize, centercrop
        unique_frames_prepared_u8, params = self._eval_prepare(
                unique_frames_rgb_u8, self.initial_resize,
                self.input_size)
        meta = Eval_Meta(
            vid=vid, video_path=video['path'],
            shifts=shifts,
            unique_real_sampled_inds=unique_real_sampled_inds)
        return (unique_frames_rgb_u8, unique_frames_prepared_u8,
                stacked_relative_sampled_inds,
                stacked_targets, params, meta)

    def get_item(self, vid: str) -> Eval_Item:
        tw = TimersWrap(['get_unique_frames', 'prepare_inputs'])
        tw.tic('get_unique_frames')
        meta: Eval_Meta
        params: Eval_Transform_Params
        (unique_frames_rgb_u8, unique_frames_prepared_u8,
                stacked_relative_sampled_inds,
                stacked_targets, params, meta) = \
                self._presample_evaluation_frames(vid)
        tw.toc('get_unique_frames')
        tw.tic('prepare_inputs')
        X = unique_frames_prepared_u8[stacked_relative_sampled_inds]
        X_tensor = torch.from_numpy(X)
        tw.toc('prepare_inputs')
        meta.tw = tw
        if self.params_to_meta:
            meta.params = params
            meta.rel_frame_inds = stacked_relative_sampled_inds
        eval_item = Eval_Item(
                X=X_tensor,
                X_plus=None,
                stacked_targets=stacked_targets,
                meta=meta)
        return eval_item


# Exotic data access variants


def _cut_box_and_reapply_training_transforms(
        box, frames_rgb, params: Train_Transform_Params, input_size):
    # wrt 256-resized image, before rcrop
    ltrd_1 = tfm_uncrop_box(box, params['rcrop'])
    # wrt original image, before scale
    ltrd_0 = tfm_unresize_box(ltrd_1, params['resize'])
    l_, t_, r_, d_ = ltrd_0
    # Cut, resize, normalize
    X_plus = frames_rgb[:, l_:r_, t_:d_, :]
    X_plus = threaded_ocv_resize_clip(
            X_plus, (input_size, input_size))
    if params['flip']['perform']:
        X_plus = np.flip(X_plus, axis=2).copy()
    assert X_plus.dtype is np.dtype('uint8'), 'must be uin8'
    return X_plus


def _cut_box_and_reapply_evaluation_transforms(
        box, frames_rgb, params: Eval_Transform_Params, input_size):
    # wrt 256-resized image, before ccrop
    ltrd_1 = tfm_uncrop_box(box, params['ccrop'])
    # wrt original image, before scale
    ltrd_0 = tfm_unresize_box(ltrd_1, params['resize'])
    l_, t_, r_, d_ = ltrd_0
    # Cut, resize, normalize
    X_plus = frames_rgb[:, l_:r_, t_:d_, :]
    X_plus = threaded_ocv_resize_clip(
            X_plus, (input_size, input_size))
    assert X_plus.dtype is np.dtype('uint8'), 'must be uin8'
    return X_plus


class DataAccess_plus_transformed_train(DataAccess_Train):
    """Data access that returns also a transformed version right away"""
    transform_kind: str
    att_crop: Optional[int]

    def __init__(
            self, dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta, new_target,
            transform_kind, att_crop=None):
        super().__init__(dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta, new_target)
        self.transform_kind == transform_kind
        self.att_crop = att_crop

    def get_item(self, vid, shift=None) -> Train_Item:
        meta: Train_Meta
        target, meta, frames_rgb_u8 = \
                self._presample_training_frames(vid, shift)
        X, params = self._apply_training_transforms(frames_rgb_u8,
                self.initial_resize, self.input_size)
        X_tensor = torch.from_numpy(X)
        # // Second 64 frames
        if self.transform_kind == 'mirror':
            X_plus = X.copy()
        elif self.transform_kind in ['centercrop', 'randomcrop']:
            if self.transform_kind == 'centercrop':
                plus_box = _get_centerbox(
                        self.input_size, self.input_size,
                        self.att_crop, self.att_crop)
            elif self.transform_kind == 'randomcrop':
                plus_box = _get_randombox(
                        self.input_size, self.input_size,
                        self.att_crop, self.att_crop)
            if self.params_to_meta:
                meta.plus_box = plus_box
            X_plus = _cut_box_and_reapply_training_transforms(
                    plus_box, frames_rgb_u8, params, self.input_size)
        else:
            raise NotImplementedError
        X_plus_tensor = torch.from_numpy(X_plus)
        if self.params_to_meta:
            meta.params = params
        train_item = Train_Item(
                X=X_tensor,
                X_plus=X_plus_tensor,
                target=target,
                meta=meta)
        return train_item


class DataAccess_plus_transformed_eval(DataAccess_Eval):
    transform_kind: str
    att_crop: Optional[int]

    def __init__(
            self, dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta, new_target, eval_gap,
            transform_kind, att_crop=None):
        super().__init__(dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta, new_target)
        self.transform_kind == transform_kind
        self.att_crop = att_crop

    def get_item(self, vid) -> Eval_Item:
        meta: Eval_Meta
        tw = TimersWrap(['get_unique_frames', 'prepare_inputs'])
        tw.tic('get_unique_frames')
        (unique_frames_rgb_u8, unique_frames_prepared_u8,
                stacked_relative_sampled_inds,
                stacked_targets, params, meta) = \
                self._presample_evaluation_frames(vid)
        tw.toc('get_unique_frames')
        tw.tic('prepare_inputs')
        X = unique_frames_prepared_u8[stacked_relative_sampled_inds]
        X_tensor = torch.from_numpy(X)
        # // Second 64 frames
        if self.transform_kind == 'mirror':
            unique_frames_plus = unique_frames_prepared_u8.copy()
        elif self.transform_kind in ['centercrop', 'randomcrop']:
            if self.transform_kind == 'centercrop':
                plus_box = _get_centerbox(
                        self.input_size, self.input_size,
                        self.att_crop, self.att_crop)
            elif self.transform_kind == 'randomcrop':
                plus_box = _get_randombox(
                        self.input_size, self.input_size,
                        self.att_crop, self.att_crop)
            if self.params_to_meta:
                meta.plus_box = plus_box
            unique_frames_plus = _cut_box_and_reapply_evaluation_transforms(
                    plus_box, unique_frames_rgb_u8,
                    params, self.input_size)
        else:
            raise NotImplementedError
        X_plus = unique_frames_plus[stacked_relative_sampled_inds]
        X_plus_tensor = torch.from_numpy(X_plus)
        tw.toc('prepare_inputs')
        meta.tw = tw
        if self.params_to_meta:
            meta.params = params
            meta.rel_frame_inds = stacked_relative_sampled_inds
        eval_item = Eval_Item(
                X=X_tensor,
                X_plus=X_plus_tensor,
                stacked_targets=stacked_targets,
                meta=meta)
        return eval_item
