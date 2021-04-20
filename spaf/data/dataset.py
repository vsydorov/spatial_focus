"""
Basic classes that allow working with external datasets
"""
import hashlib
import csv
import re
import logging
import concurrent.futures
from abc import abstractmethod, ABC
from pathlib import Path
from typing import (  # NOQA
        Dict, List, Tuple, cast, NewType, Any, Optional, TypedDict)

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import torch

from spaf.data.video import (OCV_rstats, video_capture_open, video_sample)

log = logging.getLogger(__name__)


Charades_action = TypedDict('Charades_action', {
    'name': str,
    'start': float,
    'end': float
    })
Video_charades = TypedDict('Video_charades', {
    'vid': str,
    'subject': str,
    'scene': str,
    'quality': Optional[int],
    'relevance': Optional[int],
    'verified': str,
    'script': str,
    'objects': List[str],
    'descriptions': str,
    'action_names': List[str],
    'actions': List[Charades_action],
    'length': float,
    # These are set later
    'path': Path,
    'split': str,
    'ocv_stats': OCV_rstats,
})


def charades_read_names(fold_annotations):
    # Load classes
    with (fold_annotations/'Charades_v1_classes.txt').open('r') as f:
        lines = f.readlines()
    action_names = [x.strip()[5:] for x in lines]
    # Load objects
    with (fold_annotations/'Charades_v1_objectclasses.txt').open('r') as f:
        lines = f.readlines()
    object_names = [x.strip()[5:] for x in lines]
    # Load verbs
    with (fold_annotations/'Charades_v1_verbclasses.txt').open('r') as f:
        lines = f.readlines()
    verb_names = [x.strip()[5:] for x in lines]
    return action_names, object_names, verb_names


def charades_read_video_csv(
        csv_filename, charades_action_names) -> List[Video_charades]:
    videos = []
    with csv_filename.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            objects = row['objects'].split(';')
            actions: List[Charades_action]
            if len(row['actions']) == 0:
                actions = []
                action_names = []
            else:
                actions = []
                for _action in row['actions'].split(';'):
                    vid, beg, end = _action.split()
                    action_name = charades_action_names[int(vid[1:])]
                    actions.append({
                        'name': action_name,
                        'start': float(beg),
                        'end': float(end)})
                action_names = list(set([a['name'] for a in actions]))
            relevance = int(row['relevance']) if row['relevance'] else None
            quality = int(row['quality']) if row['quality'] else None
            video: Video_charades = {
                    'vid': row['id'],
                    'subject': row['subject'],
                    'scene': row['scene'],
                    'quality': quality,
                    'relevance': relevance,
                    'verified': row['verified'],
                    'script': row['script'],
                    'objects': objects,
                    'descriptions': row['descriptions'],
                    'action_names': action_names,
                    'actions': actions,
                    'length': float(row['length']),
                    'path': None,  # type: ignore
                    'split': None,  # type: ignore
                    'ocv_stats': None,  # type: ignore
                    }
            videos.append(video)
        return videos

class Dataset_charades(object):
    """
    Charades dataset

    - This features the "trainval" split that has annotations released
    """
    videos: Dict[str, Video_charades]

    def __init__(self, action_names, object_names, verb_names, videos):
        self.action_names = action_names
        self.object_names = object_names
        self.verb_names = verb_names
        self.videos = videos


def _sample_shifted_window_dense(
        window_size: int, total_frames: int, shift: float):
    """
    Densely sample "window_size" frames, at "shift" relative position

    - "shift" between 0 and 1
    - "window_size" is bigger than "total_frames" - do edge padding

    Returns: indices of sampled frames
    """
    n_padded = max(0, window_size - total_frames)
    start_segment = total_frames - window_size + n_padded
    shift_segment = int(shift * start_segment)
    sampled_inds = shift_segment + np.arange(window_size-n_padded)
    sampled_inds = np.pad(sampled_inds, [0, n_padded], mode='edge')
    return sampled_inds


def _sample_shifted_window_fps(
        window_size: int, nframes: int,
        length_in_seconds: float, shift: float, fps: float):
    """
    Sample "window_size" frames, at "shift" r.position, at "fps"

    - "shift" between 0 and 1
    - "length" in seconds

    Returns: sampled frames inds, times
    """
    # Video frames to query
    real_fps = nframes/length_in_seconds
    if fps is None:
        # Use video fps, everything is straightforwards
        good_inds = np.arange(nframes)
    else:
        n_fake_frames = int(length_in_seconds*fps)
        fake_inds = np.interp(
                np.linspace(0, 1, n_fake_frames),
                np.linspace(0, 1, nframes),
                np.arange(nframes))
        good_inds = fake_inds.round().astype(int)

    sampled_inds = _sample_shifted_window_dense(
            window_size, len(good_inds), shift)
    real_sampled_inds = good_inds[sampled_inds]
    sampled_times = real_sampled_inds/real_fps
    return real_sampled_inds, sampled_times


class Sampler_charades(object):
    # Replacement for SA
    dataset: Dataset_charades
    labels_present = True

    def __init__(self, dataset: Dataset_charades):
        self.dataset = dataset

    def get_video(self, vid: str) -> Video_charades:
        video = self.dataset.videos[vid]
        return video

    def sample_frameids_and_times(
            self, video: Video_charades,
            shift: float, window_size: int, fps: int):
        nframes = video['ocv_stats']['max_pos_frames']
        length_in_seconds = video['length']

        real_sampled_inds, sampled_times = _sample_shifted_window_fps(
                window_size, nframes, length_in_seconds, shift, fps)
        return real_sampled_inds, sampled_times

    def sample_frames(
            self, video: Video_charades,
            real_sampled_inds: np.ndarray) -> np.ndarray:
        video_path = video['path']
        with video_capture_open(video_path, np.inf) as vcap:
            frames_u8 = np.array(video_sample(vcap, real_sampled_inds))
        return frames_u8

    def sample_targets(
            self, video: Video_charades,
            sampled_times: np.ndarray) -> torch.Tensor:
        """ Samples indices of charades actions at times """
        action_names = self.dataset.action_names
        num_classes = len(action_names)
        video_actions = video['actions']
        tars = []
        for sampled_time in sampled_times:
            target = torch.IntTensor(num_classes).zero_()
            for action in video_actions:
                if action['start'] < sampled_time < action['end']:
                    action_id = action_names.index(action['name'])
                    target[action_id] = 1
            tars.append(target)
        target = torch.stack(tars)
        return target


def list_to_hash(
        lst: List[str],
        hashlen=10) -> str:
    h = hashlib.blake2b(digest_size=hashlen)  # type: ignore
    for l in lst:
        h.update(l.encode())
    return h.hexdigest()

def _print_split_stats(self):
    # Little bit of stats
    split = self.split
    split_id = self.split_id
    s = pd.Series(split)
    svc = s.value_counts()
    if 'train' in svc and 'val' in svc:
        svc['trainval'] = svc['train'] + svc['val']
    log.info('Split {}'.format(split_id))
    log.info('Split value counts:\n{}'.format(svc))
    if 'val' in svc:
        vids_val = s[s == 'val'].index
        val_hash = self.list_to_hash(vids_val.tolist())
        log.info(f'Validation subset hash {split_id}: {val_hash}')
