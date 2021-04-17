import numpy as np
import copy
import logging

import torch

import vst

from spaf import gs_model

log = logging.getLogger(__name__)


def to_gpu_and_normalize_external(X, norm_mean, norm_std):
    X_f32c = X.type(
            torch.cuda.FloatTensor, non_blocking=True)
    X_f32c /= 255
    X_f32c = (X_f32c-norm_mean)/norm_std
    return X_f32c


def explicit_unroll_time(a, target, training):
    if training:
        nc = a.shape[2]

        # max over time, and add it to the batch
        a_video = a.mean(dim=1)
        target_video = target.max(dim=1)[0]

        # upsample a in temporal dimension if it is smaller than target (I3D)
        a = torch.nn.functional.interpolate(
            a.permute(0, 2, 1), target.shape[1],
            mode='linear', align_corners=True).permute(0, 2, 1)

        # unroll over time
        a = a.contiguous().view(-1, nc)
        target = target.contiguous().view(-1, nc)

        # combine both
        a = torch.cat([a, a_video])
        target = torch.cat([target, target_video])
    else:
        a = a.mean(dim=1)
        target = target.max(dim=1)[0]
    return a, target


def _bce_train_criterion(output, target):
    output_ups, target_ups = \
            explicit_unroll_time(output, target, True)
    target_ups = target_ups.float()
    loss = torch.nn.BCEWithLogitsLoss()(
            output_ups, target_ups.float())
    return output_ups, loss, target_ups


def _bce_eval_criterion(output, target):
    output_ups, target_ups = \
            explicit_unroll_time(output, target, False)
    target_ups = target_ups.float()
    loss = torch.nn.BCEWithLogitsLoss()(
            output_ups, target_ups.float())
    return output_ups, loss, target_ups


def load_my_checkpoint(model, optimizer, checkpoint_path):
    states = torch.load(checkpoint_path)
    model.load_state_dict(states['model_sdict'])
    optimizer.load_state_dict(states['optimizer_sdict'])
    start_epoch = states['epoch']
    return start_epoch


def checkpoints_save(rundir, epoch, model, optimizer):
    # model_{epoch} - "after epoch was finished"
    save_filepath = \
            "{}/model_at_epoch_{:03d}.pth.tar".format(rundir, epoch)
    states = {
        'epoch': epoch,
        'model_sdict': model.state_dict(),
        'optimizer_sdict': optimizer.state_dict(),
    }
    with vst.QTimer() as qtr:
        torch.save(states, str(save_filepath))
    log.info('Saved model. Epoch {}, Path {}. Took {:.2f}s'.format(
        epoch, save_filepath, qtr.time))


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


def adjust_learning_rate_explicitly(startlr, decay_rate_str, optimizer, epoch):
    if decay_rate_str is None:
        decay_rates = []
    else:
        decay_rates = [int(x) for x in decay_rate_str.split(',')]
    lr = startlr
    for d in decay_rates:
        if epoch >= d:
            lr *= 0.1
    log.info('Adjusted lr = {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class NWrap(object):
    _model = None
    _optimizer = None

    def __init__(self, model, model_type, optimizer,
            lr, lr_decay_rate, eval_batch_size):
        self._model = model
        self.model_type = model_type
        self._optimizer = optimizer
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.eval_batch_size = eval_batch_size

    def lr_epoch_adjustment(self, epoch):
        adjust_learning_rate_explicitly(
                self.lr, self.lr_decay_rate,
                self._optimizer, epoch)

    def load_my_checkpoint(self, checkpoint_path):
        start_epoch = load_my_checkpoint(
                self._model, self._optimizer, checkpoint_path)
        return start_epoch

    def load_gunnar_pretrained(self):
        path = gs_model.get_gunnar_pretrained_path(self.model_type)
        gs_model.checkpoints_load_simplified(self._model, path)

    def load_gunnar_checkpoint(self, checkpoint_path):
        gs_model.checkpoints_load_simplified(self._model, checkpoint_path)

    def forward_model(self, X):
        Y = self._model(X, None)
        return Y

    def train_criterion(self, output, target):
        output_ups, loss, target_ups = _bce_train_criterion(
                output, target)
        return output_ups, loss, target_ups

    def eval_criterion(self, output, target):
        output_ups, loss, target_ups = _bce_eval_criterion(
                output, target)
        return output_ups, loss, target_ups

    def forward_by_eval_batch_size(self, input_):
        indices = np.arange(len(input_))
        split_indices = vst.leqn_split(indices, self.eval_batch_size)
        output = []
        for split_ii in split_indices:
            # Send to cuda
            split_input_c = input_[split_ii].cuda()
            # Output stuff
            with torch.no_grad():
                split_output = self.forward_model(split_input_c)
            output.append(split_output)
        output = torch.cat(output, dim=0)
        return output

    def forward_by_eval_batch_size_efficient(self, X, norm_mean, norm_std):
        indices = np.arange(len(X))
        split_indices = vst.leqn_split(indices, self.eval_batch_size)
        output = []
        for split_ii in split_indices:
            # Split
            split_input = X[split_ii]
            # Send to cuda
            split_f32c = to_gpu_and_normalize_external(
                    split_input, norm_mean, norm_std)
            # Output stuff
            with torch.no_grad():
                split_output = self.forward_model(split_f32c)
            output.append(split_output)
        output = torch.cat(output, dim=0)
        return output

    def checkpoints_save(self, rundir, epoch):
        checkpoints_save(rundir, epoch,
                self._model, self._optimizer)

    def get_state_dicts(self):
        state_dicts = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict()}
        return state_dicts

    def load_state_dicts(self, state_dicts):
        self._model.load_state_dict(
                state_dicts['model'])
        self._optimizer.load_state_dict(
                state_dicts['optimizer'])

    def set_train(self):
        self._model.train()
        self._optimizer.zero_grad()

    def set_eval(self):
        self._model.eval()

    def optimizer_step(self):
        self._optimizer.step()
        self._optimizer.zero_grad()
