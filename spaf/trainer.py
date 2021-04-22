from __future__ import annotations
import re
import shutil
import numpy as np
import logging
from pathlib import Path
from functools import partial
import typing
from typing import Dict, Optional

import torch

import vst

from spaf.utils import (enforce_all_seeds, get_period_actions)
from spaf.network_wrap import (mkinds_to_string)
if typing.TYPE_CHECKING:
    from spaf.network_wrap import (TModel_wrap, Networks_wrap_single)
    from spaf.batcher import (Batcher_train_basic, Batcher_eval_basic)

log = logging.getLogger(__name__)


def batch_and_cache_vids(folder, vids, size_vidbatch, shuffle=True):
    METABATCHNAME = f'metabatch_S{size_vidbatch}.pkl'

    # Shuffle vids, divide into batches
    metabatch_file = folder/METABATCHNAME
    if metabatch_file.exists():
        batches_of_vids = vst.load_pkl(metabatch_file)
    else:
        if shuffle is True:
            vids_ = np.random.permutation(vids)
            log.debug('Vids shuffled to {}'.format(vids_))
        else:
            vids_ = vids
        batches_of_vids = vst.leqn_split(vids_, size_vidbatch)
        vst.save_pkl(metabatch_file, batches_of_vids)
    return batches_of_vids

def worker_killed_by_signal(e):
    return bool(re.match(r"DataLoader worker .* killed by signal", str(e)))

def signal_kill_restart(func_batcher, num_attempts=99):
    for i_attempt in range(num_attempts):
        try:
            return func_batcher()
        except RuntimeError as e:
            SIGNAL_KILL = worker_killed_by_signal(e)
            SIGNAL_KILL |= 'exited unexpectedly' in str(e) and worker_killed_by_signal(e.__cause__)
            if SIGNAL_KILL:
                log.info(f'Caught signal kill "{e}", restarting batcher, attempt {i_attempt}/{num_attempts}')
            else:
                log.info('Caught different signal, attempt {i_attempt}/{num_attempts}, raising')
                raise e
    raise RuntimeError(f'Too many batcher restart attempts {i_attempt}')


class Trainer(object):
    rundir: Path
    nswrap: Networks_wrap_single
    batcher_train: Batcher_train_basic
    batcher_eval: Batcher_eval_basic

    def __init__(self, rundir: Path,
            nswrap: Networks_wrap_single,
            batcher_train, batcher_eval,
            size_vidbatch_train, size_vidbatch_eval):

        self.rundir = rundir
        self.nswrap = nswrap
        self.batcher_train = batcher_train
        self.batcher_eval = batcher_eval
        self.size_vidbatch_train = size_vidbatch_train
        self.size_vidbatch_eval = size_vidbatch_eval

        self.train_dir = rundir/'TRAIN'
        self.eval_dir = rundir/'EVAL'

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
            self.nswrap.lr_epoch_adjustment(epoch)
            period_actions = get_period_actions(epoch, period_specs)
            # Epoch seed
            if epoch_seed:
                enforce_all_seeds(manual_seed+epoch)
            # Train
            log.info('Epoch {:03d}. Action: train'.format(epoch))
            self.train_epoch(train_vids, epoch, total_epochs)
            # Saving model
            if period_actions['checkpoint']:
                self.nswrap.checkpoints_save(self.rundir, epoch)
            if enable_eval:
                datanames_to_eval = [
                    k for k in ['qeval', 'eval'] if period_actions[k]]
                subloop_eval_vids_dict = {k: eval_vids_dict[k]
                        for k in datanames_to_eval}
                self.evaluation_subloop(
                        epoch, subloop_eval_vids_dict)

    def train_epoch(self, vids, epoch, total_epochs) -> None:
        batcher_folder = vst.mkdir(self.train_dir/f'train_{epoch:03d}')
        self.purge_old(self.train_dir, r'train_(\d{3})')

        self.nswrap.set_train()

        batches_of_vids = batch_and_cache_vids(
                batcher_folder, vids, self.size_vidbatch_train, shuffle=True)

        def func_batcher():
            return self.batcher_train.execute_epoch(
                    batches_of_vids, batcher_folder, epoch)

        train_meters = signal_kill_restart(func_batcher)

        # # Stats
        # train_meters_str = ' '.join(['{}: {m.avg:.4f}'.format(
        #         k, m=train_meters[k]) for k in ['loss', 'acc1', 'acc5']])
        # log.info(('== Results after epoch {}/{} ==\n'
        #     '\tTrain {}') .format(epoch, total_epochs,
        #         train_meters_str))

    def eval_epoch(self, vids, suffix, epoch) -> None:
        batcher_folder = vst.mkdir(
                self.eval_dir/'eval_{}_{:03d}'.format(suffix, epoch))

        self.nswrap.set_eval()

        batches_of_vids = batch_and_cache_vids(
                batcher_folder, vids, self.size_vidbatch_eval, shuffle=False)

        def func_batcher():
            return self.batcher_eval.execute_epoch(
                    batches_of_vids, batcher_folder, epoch)

        output_items = signal_kill_restart(func_batcher)
        results_mkinds = self.nswrap.outputs_items_to_results(output_items)

        # Stats
        metrics_str = mkinds_to_string(results_mkinds)
        log.info(('==={} set results at epoch {}===:\n{}').format(
            suffix, epoch, metrics_str))
