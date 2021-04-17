from __future__ import annotations
import re
import shutil
import numpy as np
import logging
from pathlib import Path
from functools import partial
from dataclasses import dataclass, asdict
import typing
from typing import Dict, Optional

import torch

import vst

from spaf.utils import (enforce_all_seeds, get_period_actions)
from spaf.network_wrap import (NWrap)
if typing.TYPE_CHECKING:
    from spaf.batcher import (TrainMetaBatcher, EvalMetaBatcher)

log = logging.getLogger(__name__)


@dataclass
class Metrics_Charades:
    mAP: float=np.nan
    acc1: float=np.nan
    acc5: float=np.nan
    loss: float=np.nan
    tloss: float=np.nan


MKinds_Charades = Dict[str, Metrics_Charades]


def metrics_to_string(m: Metrics_Charades) -> str:
    metrics_str = ' '.join((
            'mAP: {mAP:.5f}',
            'acc1: {acc1:.5f}',
            'acc5: {acc5:.5f}',
            'loss: {loss:.5f}',
            'tloss: {tloss:.5f}'
            )).format(**asdict(m))
    return metrics_str


def mkinds_to_string(mk: MKinds_Charades, join_char: str = '\n') -> str:
    # Divide outputs into 3 kinds
    metrics_strs = {}
    for kind, m in mk.items():
        metrics_str = metrics_to_string(m)
        metrics_strs[kind] = metrics_str
    metrics_superstr = join_char.join([f'{k}: {v}'
        for k, v in metrics_strs.items()])
    return metrics_superstr


def batch_and_cache_vids(self, folder, vids, shuffle=True):
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


class Trainer(object):
    rundir: Path
    nwrap: NWrap
    batcher_train: TrainMetaBatcher
    batcher_eval: EvalMetaBatcher

    def __init__(self, rundir, nwrap,
            batcher_train, batcher_eval):

        self.rundir = rundir
        self.nwrap = nwrap
        self.batcher_train = batcher_train
        self.batcher_eval = batcher_eval

        self.train_dir = rundir/'TRAIN'
        self.eval_dir = rundir/'EVAL'

    def restore_model_magic(
            self, checkpoint_path, inputs_model, training_start_epoch):
        # Resume/Load model
        if checkpoint_path:
            start_epoch = self.nwrap.load_my_checkpoint(checkpoint_path)
            start_epoch += 1
            log.info('Continuing training from checkpoint {}. '
                    'Epoch {} (ckpt + 1)'.format(checkpoint_path, start_epoch))
        else:
            if inputs_model is None:
                self.nwrap.load_gunnar_pretrained()
                log.info('Loaded model from Gunnar')
            else:
                try:
                    # First try loading my model
                    self.nwrap.load_my_checkpoint(inputs_model)
                    log.info('Loaded my model from '
                            'checkpoint {}'.format(inputs_model))
                except KeyError:
                    # Then try gunnars loading
                    self.nwrap.load_gunnar_checkpoint(inputs_model)
                    log.info('Loaded gunnars model from '
                            'checkpoint {}'.format(inputs_model))

            start_epoch = training_start_epoch
            log.info('Setting start epoch at {}'.format(start_epoch))
        return start_epoch

    def eval_check_after_restore(
            self, checkpoint_path, enable_eval, epoch, period_specs):
        """
        If model was restored from checkpoint - check if we had an evaluation
        scheduled for previous epoch and execute it.
        This covers the case of evaluation interruption after model save
        """
        period_actions = get_period_actions(
                epoch - 1, period_specs)
        datanames_to_eval = [
                k for k in ['qeval', 'eval'] if period_actions[k]]
        if checkpoint_path and enable_eval and len(datanames_to_eval):
            log.info('Rerunning scheduled evaluation.')
            self.evaluation_subloop(epoch - 1, datanames_to_eval)

    @staticmethod
    def purge_old(rundir, pattern):
        intermediate = {}
        for filename in rundir.iterdir():
            matched = re.match(pattern, filename.name)
            if matched:
                intermediate[int(matched.group(1))] = filename
        inds_to_purge = np.sort(np.fromiter(
            intermediate.keys(), np.int))[:-2]
        for ind in inds_to_purge:
            filename = intermediate[ind]
            shutil.rmtree(str(filename))

    def evaluation_subloop(self, epoch, subloop_eval_vids_dict):
        # Evaluating model
        for k, vids in subloop_eval_vids_dict.items():
            log.info('EVALUATING: Epoch {:03d}. DATA: {}'.format(
                epoch, k))
            self.eval_epoch(vids, k, epoch)

    def train_loop(
            self, train_vids, eval_vids_dict,
            start_epoch, total_epochs,
            epoch_seed: bool, manual_seed: int,
            enable_eval: bool,
            period_specs):

        for epoch in range(start_epoch, total_epochs):
            self.nwrap.lr_epoch_adjustment(epoch)
            period_actions = get_period_actions(epoch, period_specs)
            # Epoch seed
            if epoch_seed:
                enforce_all_seeds(manual_seed+epoch)
            # Train
            log.info('Epoch {:03d}. Action: train'.format(epoch))
            self.train_epoch(train_vids, epoch, total_epochs)
            # Saving model
            if period_actions['checkpoint']:
                self.nwrap.checkpoints_save(self.rundir, epoch)
            if enable_eval:
                datanames_to_eval = [
                    k for k in ['qeval', 'eval'] if period_actions[k]]
                subloop_eval_vids_dict = {k: eval_vids_dict[k]
                        for k in datanames_to_eval}
                self.evaluation_subloop(
                        epoch, subloop_eval_vids_dict)

    def train_epoch(self, vids, epoch, total_epochs):
        batcher_folder = vst.mkdir(self.train_dir/f'train_{epoch:03d}')
        self.purge_old(self.train_dir, r'train_(\d{3})')

        self.nwrap.set_train()

        batches_of_vids = batch_and_cache_vids(
                batcher_folder, vids, shuffle=True)
        train_meters = self.batcher_train.execute_epoch(
                batches_of_vids, batcher_folder, epoch)
        # Stats
        train_meters_str = ' '.join(['{}: {m.avg:.4f}'.format(
                k, m=train_meters[k]) for k in ['loss', 'acc1', 'acc5']])
        log.info(('== Results after epoch {}/{} ==\n'
            '\tTrain {}') .format(epoch, total_epochs,
                train_meters_str))

    def eval_epoch(self, vids, suffix, epoch):
        batcher_folder = vst.mkdir(
                self.eval_dir/'eval_{}_{:03d}'.format(suffix, epoch))

        self.nwrap.set_eval()

        results: MKinds_Charades = self.batcher_eval.run(
                vids, batcher_folder, epoch, suffix)
        # Stats
        metrics_str = mkinds_to_string(results)
        log.info(('==={} set results at epoch {}===:\n{}').format(
            suffix, epoch, metrics_str))
