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
        TF_params_resize, TF_params_crop, TF_params_flip,
        threaded_ocv_resize_clip,
        _get_centerbox, _get_randombox,
        tfm_uncrop_box, tfm_unresize_box,
        TimersWrap)

from spaf.data.dataset import (
        Sampler_charades, Video_charades)

log = logging.getLogger(__name__)


class TF_params_grouped(TypedDict):
    resize: TF_params_resize
    crop: TF_params_crop
    flip: Optional[TF_params_flip]


@dataclass
class Train_Meta:
    vid: str
    video_path: Path
    real_sampled_inds: np.ndarray
    plus_box: Optional[np.ndarray] = None
    params: Optional[TF_params_grouped] = None


@dataclass
class Eval_Meta:
    vid: str
    video_path: Path
    shifts: np.ndarray
    unique_real_sampled_inds: np.ndarray
    plus_box: Optional[np.ndarray] = None
    rel_frame_inds: Optional[np.ndarray] = None
    params: Optional[TF_params_grouped] = None
    tw: Optional[TimersWrap] = None


class Train_Item(TypedDict):
    X: torch.Tensor
    X_plus: Optional[torch.Tensor]
    target: torch.Tensor
    train_meta: Train_Meta


class Train_Item_collated(TypedDict):
    X: torch.Tensor
    X_plus: Optional[torch.Tensor]
    target: torch.Tensor
    train_meta: List[Train_Meta]


class Eval_Item(TypedDict):
    X: torch.Tensor   # + batch dimension
    X_plus: Optional[torch.Tensor]  # + batch dimension
    stacked_targets: torch.Tensor
    eval_meta: Eval_Meta


class DataAccess(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_item(self, vid: str):
        raise NotImplementedError()


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
            train_gap, fps, params_to_meta):
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
            ) -> Tuple[np.ndarray, TF_params_grouped]:
        X, resize_params = tfm_video_resize_threaded(
                X, initial_resize)
        X, rcrop_params = tfm_video_random_crop(
                X, input_size, input_size)
        X, flip_params = tfm_maybe_flip(X)
        assert X.dtype is np.dtype('uint8'), 'must be uin8'
        params: TF_params_grouped = {
                'resize': resize_params,
                'crop': rcrop_params,
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

        train_meta = Train_Meta(
            vid=vid, video_path=video['path'],
            real_sampled_inds=real_sampled_inds)
        return target, train_meta, frames_u8

    def get_item(self, vid: str, shift=None) -> Train_Item:
        train_meta: Train_Meta
        target, train_meta, frames_rgb_u8 = \
                self._presample_training_frames(vid, shift)
        X, params = self._apply_training_transforms(frames_rgb_u8,
                self.initial_resize, self.input_size)
        X_tensor = torch.from_numpy(X)
        if self.params_to_meta:
            train_meta.params = params
        train_item = Train_Item(
                X=X_tensor,
                X_plus=None,
                target=target,
                train_meta=train_meta)
        return train_item


class DataAccess_Eval(DataAccess):
    sampler: Sampler_charades
    initial_resize: int
    input_size: int
    train_gap: int
    fps: int
    params_to_meta: bool
    eval_gap: int

    def __init__(self, dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta, eval_gap):
        super().__init__()
        self.sampler = Sampler_charades(dataset)
        self.initial_resize = initial_resize
        self.input_size = input_size
        self.train_gap = train_gap
        self.fps = fps
        self.params_to_meta = params_to_meta
        self.eval_gap = eval_gap

    @staticmethod
    def _eval_prepare(
            X, initial_resize, input_size
            ) -> Tuple[np.ndarray, TF_params_grouped]:
        X, resize_params = tfm_video_resize_threaded(
                X, initial_resize)
        X, ccrop_params = tfm_video_center_crop(
                X, input_size, input_size)
        assert X.dtype is np.dtype('uint8'), 'must be uin8'
        params: TF_params_grouped = {
                'resize': resize_params,
                'crop': ccrop_params,
                'flip': None}
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
        eval_meta = Eval_Meta(
            vid=vid, video_path=video['path'],
            shifts=shifts,
            unique_real_sampled_inds=unique_real_sampled_inds)
        return (unique_frames_rgb_u8, unique_frames_prepared_u8,
                stacked_relative_sampled_inds,
                stacked_targets, params, eval_meta)

    def get_item(self, vid: str) -> Eval_Item:
        tw = TimersWrap(['get_unique_frames', 'prepare_inputs'])
        tw.tic('get_unique_frames')
        eval_meta: Eval_Meta
        params: TF_params_grouped
        (unique_frames_rgb_u8, unique_frames_prepared_u8,
                stacked_relative_sampled_inds,
                stacked_targets, params, eval_meta) = \
                self._presample_evaluation_frames(vid)
        tw.toc('get_unique_frames')
        tw.tic('prepare_inputs')
        X = unique_frames_prepared_u8[stacked_relative_sampled_inds]
        X_tensor = torch.from_numpy(X)
        tw.toc('prepare_inputs')
        eval_meta.tw = tw
        if self.params_to_meta:
            eval_meta.params = params
            eval_meta.rel_frame_inds = stacked_relative_sampled_inds
        eval_item = Eval_Item(
                X=X_tensor,
                X_plus=None,
                stacked_targets=stacked_targets,
                eval_meta=eval_meta)
        return eval_item


def reapply_grouped_transforms(
        frames_rgb, params: TF_params_grouped):
    # // Repeat training video preparation
    X = threaded_ocv_resize_clip(frames_rgb, params['resize']['dsize'])
    p = params['crop']
    X = X[:,
          p['i']:p['i']+p['th'],
          p['j']:p['j']+p['tw'], :]
    if (params['flip'] is not None) and params['flip']['perform']:
        X = np.flip(X, axis=2).copy()
    # At this point out shape will be input_size x input_size
    assert X.dtype is np.dtype('uint8'), 'must be uin8'
    return X


def unapply_grouped_transforms_and_cut_box(
        box, frames_rgb, params: TF_params_grouped, input_size):
    """
    input_size - dim at the entrance to CNN
    """
    # If flip was done - this is essential
    if (params['flip'] is not None) and params['flip']['perform']:
        l_, t_, r_, d_ = box
        box_ = np.r_[l_, input_size-d_, r_, input_size-t_]
    else:
        box_ = box.copy()
    # wrt 256-resized image, before rcrop
    ltrd_1 = tfm_uncrop_box(box_, params['crop'])
    # wrt original image, before scale
    ltrd_0 = tfm_unresize_box(ltrd_1, params['resize'])
    l_, t_, r_, d_ = ltrd_0
    # Cut, resize, normalize
    X_plus = frames_rgb[:, l_:r_, t_:d_, :]
    X_plus = threaded_ocv_resize_clip(
            X_plus, (input_size, input_size))
    if (params['flip'] is not None) and params['flip']['perform']:
        X_plus = np.flip(X_plus, axis=2).copy()
    assert X_plus.dtype is np.dtype('uint8'), 'must be uin8'
    return X_plus


# Exotic data access variants


def _perform_inplace_transform(
        transform_kind, input_size, att_crop, frames_rgb_u8, params):
    """
    For baselines, apply simple transform right away
    """
    if transform_kind == 'centercrop':
        plus_box = _get_centerbox(input_size, input_size, att_crop, att_crop)
    elif transform_kind == 'randomcrop':
        plus_box = _get_randombox(input_size, input_size, att_crop, att_crop)
    else:
        raise NotImplementedError
    X_plus = unapply_grouped_transforms_and_cut_box(
            plus_box, frames_rgb_u8, params, input_size)
    return X_plus, plus_box


class DataAccess_plus_transformed_train(DataAccess_Train):
    """Data access that returns also a transformed version right away"""
    transform_kind: str
    att_crop: Optional[int]

    def __init__(
            self, dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta,
            transform_kind, att_crop=None):
        super().__init__(dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta)
        self.transform_kind = transform_kind
        self.att_crop = att_crop

    def get_item(self, vid, shift=None) -> Train_Item:
        train_meta: Train_Meta
        target, train_meta, frames_rgb_u8 = \
                self._presample_training_frames(vid, shift)
        X, params = self._apply_training_transforms(frames_rgb_u8,
                self.initial_resize, self.input_size)
        X_tensor = torch.from_numpy(X)
        if self.transform_kind == 'mirror':
            X_plus = X.copy()
        else:
            X_plus, plus_box = _perform_inplace_transform(
                    self.transform_kind, self.input_size, self.att_crop,
                    frames_rgb_u8, params)
            if self.params_to_meta:
                train_meta.plus_box = plus_box
        X_plus_tensor = torch.from_numpy(X_plus)
        if self.params_to_meta:
            train_meta.params = params
        train_item = Train_Item(
                X=X_tensor,
                X_plus=X_plus_tensor,
                target=target,
                train_meta=train_meta)
        return train_item


class DataAccess_plus_transformed_eval(DataAccess_Eval):
    transform_kind: str
    att_crop: Optional[int]

    def __init__(
            self, dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta, eval_gap,
            transform_kind, att_crop=None):
        super().__init__(dataset, initial_resize, input_size,
            train_gap, fps, params_to_meta, eval_gap)
        self.transform_kind = transform_kind
        self.att_crop = att_crop

    def get_item(self, vid) -> Eval_Item:
        eval_meta: Eval_Meta
        tw = TimersWrap(['get_unique_frames', 'prepare_inputs'])
        tw.tic('get_unique_frames')
        (unique_frames_rgb_u8, unique_frames_prepared_u8,
                stacked_relative_sampled_inds,
                stacked_targets, params, eval_meta) = \
                self._presample_evaluation_frames(vid)
        tw.toc('get_unique_frames')
        tw.tic('prepare_inputs')
        X = unique_frames_prepared_u8[stacked_relative_sampled_inds]
        X_tensor = torch.from_numpy(X)
        # // Second 64 frames
        if self.transform_kind == 'mirror':
            X_plus = X.copy()
        elif self.transform_kind in ['centercrop', 'randomcrop']:
            unique_frames_plus, plus_box = _perform_inplace_transform(
                    self.transform_kind, self.input_size, self.att_crop,
                    unique_frames_rgb_u8, params)
            X_plus = unique_frames_plus[stacked_relative_sampled_inds]
        else:
            raise NotImplementedError
        X_plus_tensor = torch.from_numpy(X_plus)
        tw.toc('prepare_inputs')
        eval_meta.tw = tw
        if self.params_to_meta:
            eval_meta.params = params
            eval_meta.rel_frame_inds = stacked_relative_sampled_inds
        eval_item = Eval_Item(
                X=X_tensor,
                X_plus=X_plus_tensor,
                stacked_targets=stacked_targets,
                eval_meta=eval_meta)
        return eval_item
