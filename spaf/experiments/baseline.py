import re
import platform
import subprocess
import pprint
from pathlib import Path
from typing import (Dict)

import logging
import pandas as pd
import numpy as np
import cv2

import torch

import vst

from spaf import gs_model
from spaf.data.dataset import (
        charades_read_names, charades_read_video_csv, Dataset_charades)
from spaf.data.video import (compute_ocv_rstats)
from spaf.network_wrap import (
        Networks_wrap_single, Networks_wrap_stacked, Networks_wrap_twonet)
from spaf.data_access import (
        DataAccess_Train, DataAccess_Eval,
        DataAccess_plus_transformed_train,
        DataAccess_plus_transformed_eval)
from spaf.batcher import (
        Batcher_train_basic, Batcher_eval_basic,
        Batcher_train_attentioncrop, Batcher_eval_attentioncrop,
        Batcher_train_rescaled_attentioncrop, Batcher_eval_rescaled_attentioncrop
        )
from spaf.trainer import (Trainer)
from spaf.utils import (enforce_all_seeds, set_env, get_period_actions)

log = logging.getLogger(__name__)


DEFAULTS = """
inputs:
    dataset: ~
    model: ~
    vids_qeval: ~

model:
    type: !def ['resnet50', ['resnet50', 'nl-resnet50']]

experiment: !def ['normal', [
        'normal', 'mirror', 'mirror_twonet',
        'centercrop', 'centercrop_twonet',
        'randomcrop', 'randomcrop_twonet',
        'attentioncrop', 'attentioncrop_twonet',
        'rescaled_attentioncrop', 'rescaled_attentioncrop_twonet']]

manual_seed: 0
epoch_seed: True

period:
    checkpoint: '::1'
    eval: '5::15'
    qeval: '0::5'

utility:
    train_loop_eval: False
    debug: False
    vis_period: '::'

training:
    start_epoch: 0
    epochs: 50
    lr: 0.375
    optimizer: 'sgd'
    momentum: 0.9
    weight_decay: 1.0e-07
    lr_decay_rate: '15,40'

video:
    fps: 24

torch_data:
    trainval: False
    input_size: 224
    initial_resize: 256
    train_gap: 64
    eval_gap: 10
    num_workers: 4
    video_batch_size: 2  # train batchsize
    eval_batch_size: 5
    train_batcher_size: 200
    eval_batcher_size: 50
    cull:
        train: ~
        eval: ~
        qeval: ~

attention:
    crop: 128
    kind: !def ['hacky', ['hacky', 'entropy', 'cheating']]
    perframe_reduce: !def ['max', ['max', 'min']]
    batch_attend: False
    new_target: !def ['append', ['single', 'append']]

twonet:
    original_net_kind: !def ['fixed_old', ['fixed_old', 'new']]
    fixed_net_path: ~
"""

def find_checkpoints(rundir):
    ckpt_re = r'model_at_epoch_(?P<iteration>\d*).pth'
    checkpoints = {}
    for subfolder_item in rundir.iterdir():
        search = re.search(ckpt_re, subfolder_item.name)
        if search:
            iteration = int(search.groupdict()['iteration'])
            checkpoints[iteration] = subfolder_item
    return checkpoints


def create_model(model_type, nclass):
    if model_type == 'resnet50':
        model = gs_model.get_explicit_model_i3d(nclass)
    elif model_type == 'nl-resnet50':
        model = gs_model.get_explicit_model_i3d_nonlocal(nclass)
    else:
        raise NotImplementedError()
    log.info('Model:\n{}'.format(model))
    return model


def create_optimizer(model, optimizer, lr, momentum, weight_decay):
    assert optimizer == 'sgd'
    optimizer = torch.optim.SGD(
        model.parameters(), lr,
        momentum=momentum,
        weight_decay=weight_decay)
    return optimizer


def platform_info():
    platform_string = f'Node: {platform.node()}'
    oar_jid = subprocess.run('echo $OAR_JOB_ID', shell=True,
            stdout=subprocess.PIPE).stdout.decode().strip()
    platform_string += ' OAR_JOB_ID: {}'.format(
            oar_jid if len(oar_jid) else 'None')
    platform_string += f' System: {platform.system()} {platform.version()}'
    return platform_string


def cull_vids_fraction(vids, fraction):
    if fraction is None:
        return vids
    shuffle = fraction < 0.0
    fraction = abs(fraction)
    assert 0.0 <= fraction < 1.0
    N_total = int(len(vids) * fraction)
    if shuffle:
        culled_vids = np.random.permutation(vids)[:N_total]
    else:
        culled_vids = vids[:N_total]
    return culled_vids


def prepare_charades_vids_v2(dataset: Dataset_charades, cull_specs):
    # Prepare vids
    train_vids = [vid for vid, v in dataset.videos.items()
            if v['split'] in ['train', 'val']]
    eval_vids = [vid for vid, v in dataset.videos.items()
            if v['split'] in ['test']]
    # Cull if necessary
    if cull_specs is not None:
        train_vids = cull_vids_fraction(train_vids, cull_specs['train'])
        eval_vids = cull_vids_fraction(eval_vids, cull_specs['eval'])
    # Eval dict
    eval_vids_dict = {'eval': eval_vids}
    return train_vids, eval_vids_dict


def _assign_actions_to_checkpoints(checkpoints, period_specs):
    checkpoints_and_actions = {}
    for epoch, path in checkpoints.items():
        period_actions = get_period_actions(epoch, period_specs)
        datanames_to_eval = [
                k for k in ['qeval', 'eval'] if period_actions[k]]
        if len(datanames_to_eval):
            checkpoints_and_actions[epoch] = \
                [path.name, datanames_to_eval]
    return checkpoints_and_actions


def _train_loop(cf, period_specs, trainer: Trainer,
        train_vids, eval_vids_dict) -> None:
    # Train loop
    checkpoints = find_checkpoints(trainer.rundir)
    if len(checkpoints):
        checkpoint_path = max(checkpoints.items())[1]
    else:
        checkpoint_path = None
    start_epoch = trainer.nswrap.restore_model_magic(
            checkpoint_path, cf['inputs.model'],
            cf['training.start_epoch'], cf['model.type'])
    vst.additional_logging(trainer.rundir/'TRAIN'/f'from_{start_epoch}')
    # print platform info again
    log.info(platform_info())
    trainer.eval_check_after_restore(
            checkpoint_path, cf['utility.train_loop_eval'],
            start_epoch, period_specs)
    trainer.train_loop(
            train_vids, eval_vids_dict,
            start_epoch, cf['training.epochs'],
            cf['epoch_seed'], cf['manual_seed'],
            cf['utility.train_loop_eval'],
            period_specs)


def _eval_loop(period_specs, trainer: Trainer,
        eval_vids_dict, recheck) -> None:
    # Execute evaluation procedure inplace
    vst.additional_logging(trainer.rundir/'EVAL'/'eval')
    checkpoints: Dict[int, Path] = find_checkpoints(trainer.rundir)
    checkpoints_and_actions = _assign_actions_to_checkpoints(
            checkpoints, period_specs)
    log.info('Actions per checkpoint:\n{}'.format(
            pprint.pformat(checkpoints_and_actions)))
    epochs_to_eval = list(checkpoints_and_actions.keys())
    while len(epochs_to_eval):
        epoch = epochs_to_eval.pop(0)
        checkpoint_path = checkpoints[epoch]
        datanames_to_eval = checkpoints_and_actions[epoch][1]
        subloop_eval_vids_dict = \
                {k: eval_vids_dict[k] for k in datanames_to_eval}
        trainer.nswrap.load_my_checkpoint(checkpoint_path)
        trainer.evaluation_subloop(epoch, subloop_eval_vids_dict)
        if recheck:
            # Recheck for new good checkpoints
            checkpoints = find_checkpoints(trainer.rundir)
            checkpoints_and_actions = _assign_actions_to_checkpoints(
                    checkpoints, period_specs)
            new_checkpoints = [k for k in checkpoints_and_actions.keys()
                    if k > max(epochs_to_eval, default=epoch)]
            if len(new_checkpoints):
                log.info('New checkpoint found after recheck: {}'.format(
                    new_checkpoints))
                epochs_to_eval.extend(new_checkpoints)

def _get_data_access(cf, dataset, da_kind, plus_transform_kind=None):
    initial_resize = cf['torch_data.initial_resize']
    input_size = cf['torch_data.input_size']
    train_gap = cf['torch_data.train_gap']
    eval_gap = cf['torch_data.eval_gap']
    fps = cf['video.fps']

    if da_kind == 'normal':
        data_access_train = DataAccess_Train(
                dataset, initial_resize, input_size,
                train_gap, fps, True)
        data_access_eval = DataAccess_Eval(
                dataset, initial_resize, input_size,
                train_gap, fps, True, eval_gap)
    elif da_kind == 'plus':
        data_access_train = DataAccess_plus_transformed_train(
                dataset, initial_resize, input_size,
                train_gap, fps, False, plus_transform_kind,
                cf['attention.crop'])
        data_access_eval = DataAccess_plus_transformed_eval(
                dataset, initial_resize, input_size,
                train_gap, fps, False, eval_gap, plus_transform_kind,
                cf['attention.crop'])
    else:
        raise RuntimeError()
    return data_access_train, data_access_eval

def _get_nwswrap(cf, model, optimizer, nsw_kind):
    lr = cf['training.lr']
    lr_decay_rate = cf['training.lr_decay_rate']
    NORM_MEAN = torch.cuda.FloatTensor([0.485, 0.456, 0.406])
    NORM_STD = torch.cuda.FloatTensor([0.229, 0.224, 0.225])
    att_crop = cf['attention.crop']
    att_kind = cf['attention.kind']

    if nsw_kind == 'normal':
        nswrap = Networks_wrap_single(model, optimizer,
                NORM_MEAN, NORM_STD, lr, lr_decay_rate, att_crop, att_kind)
    elif nsw_kind == 'stacked':
        nswrap = Networks_wrap_stacked(model, optimizer,
                NORM_MEAN, NORM_STD, lr, lr_decay_rate, att_crop, att_kind)
    elif nsw_kind == 'twonet':
        assert cf['twonet.original_net_kind'] == 'fixed_old'
        nswrap = Networks_wrap_twonet(model, optimizer,
                NORM_MEAN, NORM_STD, lr, lr_decay_rate, att_crop, att_kind,
                cf['twonet.fixed_net_path'])
    else:
        raise RuntimeError()
    return nswrap

def _get_batchers(cf, nswrap, data_access_train, data_access_eval, ba_kind):
    train_batch_size = cf['torch_data.video_batch_size']  # train batchsize
    eval_batch_size = cf['torch_data.eval_batch_size']   # eval batchsize
    num_workers = cf['torch_data.num_workers']

    if ba_kind == 'normal':
        batcher_train = Batcher_train_basic(
                nswrap, data_access_train, train_batch_size, num_workers)
        batcher_eval = Batcher_eval_basic(
                nswrap, data_access_eval, eval_batch_size, num_workers)
    elif ba_kind == 'attentioncrop':
        batcher_train = Batcher_train_attentioncrop(
                nswrap, data_access_train, train_batch_size, num_workers)
        batcher_eval = Batcher_eval_attentioncrop(
                nswrap, data_access_eval, eval_batch_size, num_workers)
    elif ba_kind == 'rescaled_attentioncrop':
        batcher_train = Batcher_train_rescaled_attentioncrop(
                nswrap, data_access_train, train_batch_size, num_workers)
        batcher_eval = Batcher_eval_rescaled_attentioncrop(
                nswrap, data_access_eval, eval_batch_size, num_workers)
    else:
        raise RuntimeError()
    return batcher_train, batcher_eval


# Experiments


def train_baseline(workfolder, cfg_dict, add_args):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict)
    cfg.set_defaults_yaml(DEFAULTS)
    cf = cfg.parse()
    period_specs = cfg.without_prefix('period.')
    cull_specs = cfg.without_prefix('torch_data.cull.')
    rundir = vst.mkdir(out/'RUNDIR')

    enforce_all_seeds(cf['manual_seed'])
    set_env()

    experiment_name = cf['experiment']

    # // Data preparation
    dataset = vst.load_pkl(cf['inputs.dataset'])
    if isinstance(dataset, Dataset_charades):
        nclass = 157
    else:
        raise NotImplementedError()

    # // Create network and optimizer
    model_type = cf['model.type']
    model = create_model(model_type, nclass)
    optimizer = create_optimizer(
            model, cf['training.optimizer'], cf['training.lr'],
            cf['training.momentum'], cf['training.weight_decay'])

    debug_enabled = cf['utility.debug']

    if experiment_name == 'normal':
        data_access_train, data_access_eval = _get_data_access(cf, dataset, 'normal')
        nswrap = _get_nwswrap(cf, model, optimizer, 'normal')
        batcher_train, batcher_eval = _get_batchers(
                cf, nswrap, data_access_train, data_access_eval, 'normal')
    elif experiment_name in ['mirror', 'mirror_twonet',
            'centercrop', 'centercrop_twonet', 'randomcrop', 'randomcrop_twonet']:
        plus_transform_kind = experiment_name.split('_')[0]
        nsw_kind = 'twonet' if 'twonet' in experiment_name else 'stacked'
        data_access_train, data_access_eval = _get_data_access(
                cf, dataset, 'plus', plus_transform_kind)
        nswrap = _get_nwswrap(cf, model, optimizer, nsw_kind)
        batcher_train, batcher_eval = _get_batchers(
                cf, nswrap, data_access_train, data_access_eval, 'normal')
    elif experiment_name in ['attentioncrop', 'attentioncrop_twonet']:
        nsw_kind = 'twonet' if 'twonet' in experiment_name else 'stacked'
        data_access_train, data_access_eval = _get_data_access(cf, dataset, 'normal')
        nswrap = _get_nwswrap(cf, model, optimizer, nsw_kind)
        batcher_train, batcher_eval = _get_batchers(
                cf, nswrap, data_access_train, data_access_eval, 'attentioncrop')

    elif experiment_name in ['rescaled_attentioncrop', 'rescaled_attentioncrop_twonet']:
        nsw_kind = 'twonet' if 'twonet' in experiment_name else 'stacked'
        data_access_train, data_access_eval = _get_data_access(cf, dataset, 'normal')
        nswrap = _get_nwswrap(cf, model, optimizer, nsw_kind)
        batcher_train, batcher_eval = _get_batchers(
                cf, nswrap, data_access_train, data_access_eval, 'rescaled_attentioncrop')
    else:
        raise NotImplementedError()

    # // Create trainer
    size_vidbatch_train = cf['torch_data.train_batcher_size']
    size_vidbatch_eval = cf['torch_data.eval_batcher_size']
    trainer = Trainer(rundir, nswrap,
            batcher_train, batcher_eval,
            size_vidbatch_train, size_vidbatch_eval)

    train_vids, eval_vids_dict = prepare_charades_vids_v2(dataset, cull_specs)

    if '--eval' in add_args:
        recheck = '--recheck' in add_args
        _eval_loop(period_specs, trainer, eval_vids_dict, recheck)
    else:
        _train_loop(cf, period_specs, trainer, train_vids, eval_vids_dict)
