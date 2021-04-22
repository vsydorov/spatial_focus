import concurrent.futures
import os
import time
import collections
import datetime
import logging
import random
import re
import shutil
import string
from pathlib import Path
from typing import (Dict, TypedDict, Tuple)

import numpy as np
import cv2
# import ffmpeg

import vst

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import default_collate

log = logging.getLogger(__name__)

OCV_RESIZE_THREADS = 2


def enforce_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    rgen = np.random.default_rng(seed)
    return rgen


def set_env():
    os.environ['TORCH_HOME'] = \
            '/home/vsydorov/scratch/gpuhost7/bulk/torch_home/'
    cudnn.benchmark = False
    cudnn.enabled = False


class TimersWrap(object):
    def __init__(self, names):
        self.names = names
        self.meters = {k: vst.Averager() for k in self.names}
        self.end_times = {k: - np.NINF for k in self.names}

    def tic(self, *names):
        for name in names:
            self.end_times[name] = time.time()

    def toc(self, *names):
        for name in names:
            self.meters[name].update(time.time() - self.end_times[name])

    @property
    def time_str(self):
        time_strs = []
        for time_name in self.names:
            time_strs.append('{n}: {m.last:.2f}({m.avg:.2f})s'.format(
                n=time_name, m=self.meters[time_name]))
        time_str = 'Time['+' '.join(time_strs)+']'
        return time_str


def get_period_actions(step: int, period_specs: Dict[str, str]):
    period_actions = {}
    for action, period_spec in period_specs.items():
        period_actions[action] = vst.check_step(step, period_spec)
    return period_actions


# Video utils


def yana_size_query(X, dsize):
    # https://github.com/hassony2/torch_videovision
    def _get_resize_sizes(im_h, im_w, size):
        if im_w < im_h:
            ow = size
            oh = int(size * im_h / im_w)
        else:
            oh = size
            ow = int(size * im_w / im_h)
        return oh, ow

    if isinstance(dsize, int):
        im_h, im_w, im_c = X[0].shape
        new_h, new_w = _get_resize_sizes(im_h, im_w, dsize)
        isize = (new_w, new_h)
    else:
        assert len(dsize) == 2
        isize = dsize[1], dsize[0]
    return isize


def randint0(value):
    if value == 0:
        return 0
    else:
        return np.random.randint(value)


def yana_ocv_resize_clip(X, dsize):
    isize = yana_size_query(X, dsize)
    scaled = np.stack([
        cv2.resize(img, isize, interpolation=cv2.INTER_LINEAR) for img in X
    ])
    return scaled


def threaded_ocv_resize_clip(
        X, dsize, resize_threads=None,
        interpolation=cv2.INTER_LINEAR):
    if resize_threads is None:
        resize_threads = OCV_RESIZE_THREADS
    isize = yana_size_query(X, dsize)
    thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=resize_threads)
    futures = []
    for img in X:
        futures.append(thread_executor.submit(
            cv2.resize, img, isize,
            interpolation=interpolation))
    concurrent.futures.wait(futures)
    thread_executor.shutdown()
    scaled = np.array([x.result() for x in futures])
    return scaled


def _get_randombox(h_before, w_before, th, tw):
    i = randint0(h_before-th)
    j = randint0(w_before-tw)
    return np.r_[i, j, i+th, j+tw]


def _get_centerbox(h_before, w_before, th, tw):
    i = int((h_before-th)/2)
    j = int((w_before-tw)/2)
    return np.r_[i, j, i+th, j+tw]


""" Transforms """

class TF_params_resize(TypedDict):
    dsize: int
    h_before: int
    w_before: int
    h_resized: int
    w_resized: int

class TF_params_crop(TypedDict):
    th: int
    tw: int
    h_before: int
    w_before: int
    i: int
    j: int

class TF_params_flip(TypedDict):
    perform: bool


def tfm_video_resize_threaded(X, dsize) -> Tuple[np.ndarray, TF_params_resize]:
    # 256 resize, normalize, group,
    h_before, w_before = X.shape[1:3]
    X = threaded_ocv_resize_clip(X, dsize)
    h_resized, w_resized = X.shape[1:3]
    params: TF_params_resize = {'dsize': dsize,
            'h_before': h_before, 'w_before': w_before,
            'h_resized': h_resized, 'w_resized': w_resized}
    return X, params


def tfm_video_random_crop(
        first64, th, tw
        ) -> Tuple[np.ndarray, TF_params_crop]:
    h_before, w_before = first64.shape[1:3]
    rcrop_i = randint0(h_before - th)
    rcrop_j = randint0(w_before - tw)
    first64 = first64[:,
            rcrop_i:rcrop_i+th,
            rcrop_j:rcrop_j+tw, :]
    params: TF_params_crop = {'th': th, 'tw': tw,
            'h_before': h_before, 'w_before': w_before,
            'i': rcrop_i, 'j': rcrop_j}
    return first64, params


def tfm_video_center_crop(
        first64, th, tw
        ) -> Tuple[np.ndarray, TF_params_crop]:
    h_before, w_before = first64.shape[1:3]
    ccrop_i = int((h_before-th)/2)
    ccrop_j = int((w_before-tw)/2)
    first64 = first64[:,
            ccrop_i:ccrop_i+th,
            ccrop_j:ccrop_j+tw, :]
    params: TF_params_crop = {'th': th, 'tw': tw,
            'h_before': h_before, 'w_before': w_before,
             'i': ccrop_i, 'j': ccrop_j}
    return first64, params


def tfm_maybe_flip(first64) -> Tuple[np.ndarray, TF_params_flip]:
    perform_video_flip = np.random.random() < 0.5
    if perform_video_flip:
        first64 = np.flip(first64, axis=2).copy()
    params: TF_params_flip = {'perform': perform_video_flip}
    return first64, params


""" Reverse Transforms """


def tfm_uncrop_box(box, params: TF_params_crop):
    i, j = params['i'], params['j']
    return box + [i, j, i, j]


def tfm_unresize_box(box, params: TF_params_resize):
    real_scale_h = params['h_resized']/params['h_before']
    real_scale_w = params['w_resized']/params['w_before']
    real_scale = np.tile(np.r_[real_scale_h, real_scale_w], 2)
    box = (box / real_scale).astype(int)
    return box
