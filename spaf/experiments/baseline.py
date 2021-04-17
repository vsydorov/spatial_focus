import re
import platform
import subprocess
from pathlib import Path

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
from spaf.network_wrap import (NWrap)
from spaf.data_access import (DataAccess_Train, DataAccess_Eval)
from spaf.batcher import (TrainMetaBatcher_Normal, EvalMetaBatcher_Normal)
from spaf.trainer import (Trainer)
from spaf.utils import (enforce_all_seeds, set_env)

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
    train_gap: 64
    eval_gap: 10
    num_workers: 4
    video_batch_size: 2
    eval_batch_size: 5
    train_batcher_size: 200
    eval_batcher_size: 50
    cull:
        train: [~, ~]
        val: [~, ~]
        qeval: [~, ~]
        test: [~, ~]

attention:
    crop: 128
    kind: !def ['hacky', ['hacky', 'entropy', 'cheating']]
    perframe_reduce: !def ['max', ['max', 'min']]
    batch_attend: False
    new_target: !def ['append', ['single', 'append']]
"""


TWONET_DEFTYPE = """
twonet:
    original_net_kind: ['fixed_old', ['fixed_old', 'new']]
    fixed_net_path: [~, ~]
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


def _network_preparation(cf, dataset):
    model_type = cf['model.type']

    if isinstance(dataset, Dataset_charades):
        nclass = 157
    # elif cf['dataset.name'] == 'hmdb51':
    #     nclass = 51
    else:
        raise NotImplementedError()
    model = create_model(model_type, nclass)
    optimizer = create_optimizer(
            model,
            cf['training.optimizer'],
            cf['training.lr'],
            cf['training.momentum'],
            cf['training.weight_decay'])
    return model, model_type, optimizer


def _nwrap_preparation(cf, experiment_name,
        model, model_type, optimizer):

    lr = cf['training.lr']
    lr_decay_rate = cf['training.lr_decay_rate']
    eval_batch_size = cf['torch_data.eval_batch_size']

    if experiment_name in ['normal', 'centercrop', 'randomcrop', 'mirror']:
        nwrap = NWrap(
                model, model_type, optimizer,
                lr, lr_decay_rate, eval_batch_size)
    elif experiment_name in [
            'mirror_twonet', 'centercrop_twonet', 'randomcrop_twonet']:
        nwrap = NWrap_twonet(
                model, model_type, optimizer,
                lr, lr_decay_rate, eval_batch_size,
                original_net_kind=cf['twonet.original_net_kind'],
                fixed_net_path=cf['twonet.fixed_net_path'])
        nwrap.load_second_net()
    elif experiment_name in ['attentioncrop', 'rescaled_attentioncrop']:
        nwrap = NWrap_Attention(
                model, model_type, optimizer,
                lr, lr_decay_rate, eval_batch_size,
                att_crop=cf['attention.crop'],
                attention_kind=cf['attention.kind'])
    elif experiment_name in [
            'attentioncrop_twonet',
            'rescaled_attentioncrop_twonet']:
        nwrap = NWrap_Attention_twonet(
                model, model_type, optimizer,
                lr, lr_decay_rate, eval_batch_size,
                att_crop=cf['attention.crop'],
                attention_kind=cf['attention.kind'],
                original_net_kind=cf['twonet.original_net_kind'],
                fixed_net_path=cf['twonet.fixed_net_path'])
        nwrap.load_second_net()
    else:
        raise NotImplementedError()
    return nwrap


def _data_access_preparation_train_eval(
        cf, experiment_name, dataset, nwrap):
    initial_resize = 256
    input_size = cf['torch_data.input_size']
    train_gap = cf['torch_data.train_gap']
    eval_gap = cf['torch_data.eval_gap']
    fps = cf['video.fps']
    if experiment_name == 'normal':
        data_access_train = DataAccess_Train(
                sa, initial_resize, input_size, train_gap, fps,
                params_to_meta=False,
                new_target='single')
        data_access_eval = DataAccess_Eval(
                sa, initial_resize, input_size, train_gap, fps,
                params_to_meta=False,
                new_target='single',
                eval_gap=eval_gap)
        train_batcher_cls = TrainMetaBatcher_Normal
        eval_batcher_cls = EvalMetaBatcher_Normal
    else:
        raise NotImplementedError()

    NORM_MEAN = torch.cuda.FloatTensor([0.485, 0.456, 0.406])
    NORM_STD = torch.cuda.FloatTensor([0.229, 0.224, 0.225])

    train_batcher_size = cf['torch_data.train_batcher_size']
    eval_batcher_size = cf['torch_data.eval_batcher_size']
    debug_enabled = cf['utility.debug']
    video_batch_size = cf['torch_data.video_batch_size']
    num_workers = cf['torch_data.num_workers']

    batcher_train = train_batcher_cls(
            NORM_MEAN, NORM_STD, train_batcher_size, data_access_train,
            nwrap, video_batch_size, num_workers, debug_enabled)

    batcher_eval = eval_batcher_cls(
            NORM_MEAN, NORM_STD, eval_batcher_size, data_access_eval,
            nwrap, video_batch_size, num_workers, debug_enabled)

    return batcher_train, batcher_eval


def _set_defaults(cfg, cfg_dict):
    cfg.set_defaults(DEFAULTS)
    if cfg_dict['experiment'] in [
            'mirror_twonet',
            'centercrop_twonet',
            'randomcrop_twonet',
            'attentioncrop_twonet',
            'rescaled_attentioncrop_twonet'
            ]:
        cfg.set_deftype(TWONET_DEFTYPE)


def platform_info():
    platform_string = f'Node: {platform.node()}'
    oar_jid = subprocess.run('echo $OAR_JOB_ID', shell=True,
            stdout=subprocess.PIPE).stdout.decode().strip()
    platform_string += ' OAR_JOB_ID: {}'.format(
            oar_jid if len(oar_jid) else 'None')
    platform_string += f' System: {platform.system()} {platform.version()}'
    return platform_string


def _train_loop(cf, period_specs, trainer, checkpoint_path,
                train_vids, eval_vids_dict):
    # Train loop
    start_epoch = trainer.restore_model_magic(
            checkpoint_path, cf['inputs.model'],
            cf['training.start_epoch'])
    vst.additional_logging(trainer.rundir/'TRAIN', f'from_{start_epoch}')
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
    checkpoints = find_checkpoints(rundir)
    if len(checkpoints):
        checkpoint_path = max(checkpoints.items())[1]
    else:
        checkpoint_path = None

    experiment_name = cf['experiment']

    # // Data preparation
    # data, sa, train_vids, eval_vids_dict = \
    #         _data_preparation(cf, cull_specs)
    dataset = vst.load_pkl(cf['inputs.dataset'])

    # // Create network
    model, model_type, optimizer = _network_preparation(cf, dataset)
    # // Create nwrap
    nwrap = _nwrap_preparation(cf, experiment_name,
        model, model_type, optimizer)
    # // Data access
    batcher_train, batcher_eval = _data_access_preparation_train_eval(
            cf, experiment_name, dataset, nwrap)

    # // Create trainer
    trainer = Trainer(rundir, nwrap, batcher_train, batcher_eval)

    if '--eval' in add_args:
        recheck = '--recheck' in add_args
        _eval_loop(period_specs, trainer, rundir, eval_vids_dict, recheck)
    else:
        _train_loop(cf, period_specs, trainer, checkpoint_path,
                train_vids, eval_vids_dict)
