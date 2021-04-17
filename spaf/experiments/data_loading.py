    from pathlib import Path

import logging
import pandas as pd
import numpy as np
import cv2

import vst

from spaf.data.dataset import (
        charades_read_names, charades_read_video_csv, Dataset_charades)
from spaf.data.video import (compute_ocv_rstats)

log = logging.getLogger(__name__)


def get_action_lengths(videos):
    action_lengths = []
    for video in videos:
        for action in video['actions']:
            alen = action['end'] - action['start']
            line = [video['vid'], action['name'], alen]
            action_lengths.append(line)
    action_lengths = pd.DataFrame(
            action_lengths, columns=['vid', 'name', 'length'])
    return action_lengths

def sum_action_lengths(action_lengths, vids_subset, action_names):
    good = action_lengths.vid.apply(lambda x: x in vids_subset)
    length_per_action = action_lengths[good].groupby('name')['length'].sum()
    length_per_action = length_per_action.reindex(index=action_names).fillna(0)
    alengths = length_per_action.values
    return alengths


def sample_validation_split(
        action_lengths, action_names, vids_train,
        split_seed, split_fraction, split_samplings):

    alengths_train = sum_action_lengths(
            action_lengths, vids_train, action_names)

    rgen = np.random.default_rng(split_seed)
    # Sample fraction of vids several times
    alengths_per_sample = []
    vids_per_sample = []
    for i in range(split_samplings):
        ssize = int(len(vids_train)*split_fraction)
        vids_sampled = rgen.permutation(vids_train)[:ssize]
        alengths_sampled = sum_action_lengths(
                action_lengths, vids_sampled, action_names)
        alengths_per_sample.append(alengths_sampled)
        vids_per_sample.append(vids_sampled)
    vids_per_sample = np.vstack(vids_per_sample)
    alengths_per_sample = np.vstack(alengths_per_sample)
    fractions_per_sample = alengths_per_sample/alengths_train

    max_fraction = fractions_per_sample.min(1)
    min_fraction = fractions_per_sample.max(1)
    # Reasonable checks
    good = (max_fraction > 0) & (min_fraction < 1)
    vids_per_sample = vids_per_sample[good]
    max_fraction = max_fraction[good]
    # Select split with max_fraction closest to fraction
    ags = np.argsort(np.abs(max_fraction-split_fraction))
    vids_sampled = vids_per_sample[ags[0]]
    return vids_sampled


def precompute_dataset_charades(workfolder, cfg_dict, add_args):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict)
    cfg.set_defaults_yaml("""
    annotations: ~  #  Charades.zip
    videos: ~  #  Charades_v1/Charades_v1_480
    validation_split:
        seed: 42
        fraction: 0.2
        samplings: 20
    """)
    cf = cfg.parse()

    fold_annotations = Path(cf['annotations'])
    fold_videos = Path(cf['videos'])

    # Retrieve annotation data
    action_names, object_names, verb_names = \
            charades_read_names(fold_annotations)
    videos_train = charades_read_video_csv(
            fold_annotations/'Charades_v1_train.csv', action_names)
    videos_test = charades_read_video_csv(
            fold_annotations/'Charades_v1_test.csv', action_names)
    for v in videos_train:
        v['split'] = 'train'
    for v in videos_test:
        v['split'] = 'test'
    videos = videos_train + videos_test

    # Point to real videos, get their OCV stats
    for video in videos:
        video['path'] = fold_videos/'{}.mp4'.format(video['vid'])
        assert video['path'].exists()
    isaver = vst.isave.Isaver_threading(
            out/'save_rstats', zip([v['path'] for v in videos]),
            compute_ocv_rstats, progress='video ocv stats')
    ocv_stats = isaver.run()
    # Make sure all frames are reachable
    for x in ocv_stats:
        assert x['frame_count'] == x['max_pos_frames']
    for v, x in zip(videos, ocv_stats):
        v['ocv_stats'] = x

    # Convert to dict now
    videos_dict = {v['vid']: v for v in videos}

    # Establish validation set
    action_lengths = get_action_lengths(videos)
    vids_train = [k for k, v in videos_dict.items()
            if v['split'] == 'train']
    split_seed = cf['validation_split.seed']
    split_fraction = cf['validation_split.fraction']
    split_samplings = cf['validation_split.samplings']
    vids_val = sample_validation_split(
        action_lengths, action_names, vids_train,
        split_seed, split_fraction, split_samplings)
    for vid in vids_val:
        videos_dict[vid]['split'] = 'val'

    # Print split stats
    split_ninstances = {}
    split_lengths = {}
    split_vids = {}
    for split in ['train', 'val', 'test']:
        vids = [k for k, v in videos_dict.items()
                if v['split'] == split]
        good = action_lengths.vid.apply(lambda x: x in vids)
        good_action_lengths = action_lengths[good]
        instances_per_action = good_action_lengths.groupby(
                'name')['length'].count()
        length_per_action = good_action_lengths.groupby(
                'name')['length'].sum()
        split_ninstances[split] = instances_per_action
        split_lengths[split] = length_per_action
        split_vids[split] = vids
    split_ninstances = pd.concat(split_ninstances, axis=1)
    split_length = pd.concat(split_lengths, axis=1)
    log.debug(f'Instances per action:\n{split_ninstances.to_string()}')
    log.debug(f'Length per action:\n{split_length.to_string()}')
    log.info('Vids per split:{}'.format(
        {k: len(v) for k, v in split_vids.items()}))

    dataset = Dataset_charades(
            action_names, object_names, verb_names, videos_dict)
    vst.save_pkl(out/'dataset.pkl', dataset)
