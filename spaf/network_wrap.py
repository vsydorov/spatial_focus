import numpy as np
import copy
import logging
from dataclasses import dataclass, asdict
from typing import (Dict, Optional, TypedDict)

import torch
import torch.nn.functional as F

import vst

from spaf.gs_model import MyDataParallel
from spaf import gs_model
# from spaf.utils import (leqn_split)

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


def _map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return _map(fix, gt_array)


def np_multilabel_batch_accuracy_topk(outputs, targets, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = targets.shape[0]
    maxk = max(topk)
    topk_ids = outputs.argsort(axis=1)[:, ::-1][:, :maxk]

    correct = np.zeros_like(topk_ids)
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            correct[i, j] = targets[i, topk_ids[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(correct_k/batch_size)
    return res


def _quick_scores_from_cpudicts(cpudicts) -> Metrics_Charades:
    if len(cpudicts) == 0:
        log.warn('Trying to compute scores on empty video outputs')
        return Metrics_Charades()

    scores_, score_targets_ = zip(*[
            (x['scores'], x['score_target'])
            for x in cpudicts.values()])
    scores_ = np.array([x for x in scores_])
    score_targets_ = np.array([x for x in score_targets_])
    vscores_ = scores_.max(1)
    vscore_targets_ = score_targets_.max(1)
    mAP, wAP, ap = charades_map(vscores_, vscore_targets_)
    acc1, acc5 = np_multilabel_batch_accuracy_topk(
            vscores_, vscore_targets_, topk=(1, 5))
    loss = float(np.mean([x['loss']
        for x in cpudicts.values()]))
    train_loss = float(np.mean([x['train_loss']
        for x in cpudicts.values()]))
    return Metrics_Charades(mAP, acc1, acc5, loss, train_loss)


class TModel_wrap(object):
    """
    Shallow wrap around torch model, performs normalization
    """
    def __init__(self, model, norm_mean, norm_std):
        self.model = model
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def to_gpu_and_normalize(self, X_u8):
        X_f32c = X_u8.type(torch.cuda.FloatTensor, non_blocking=True)
        X_f32c /= 255
        X_f32c = (X_f32c-self.norm_mean)/self.norm_std
        return X_f32c

    def forward_model(self, X_u8):
        X_f32c = self.to_gpu_and_normalize(X_u8)
        output = self.model(X_f32c, None)
        return output

    def forward_model_batchwise_nograd(self, X_u8, batch_size):
        indices = np.arange(len(X_u8))
        split_indices = vst.leqn_split(indices, batch_size)
        output = []
        for split_ii in split_indices:
            # Split
            split_u8 = X_u8[split_ii]
            with torch.no_grad():
                split_output = self.forward_model(split_u8)
            output.append(split_output)
        output = torch.cat(output, dim=0)
        return output


class Item_output(TypedDict):
    output_X: torch.Tensor
    output_X_plus: Optional[torch.Tensor]
    target: torch.Tensor

def get_loss_entropy(output_0):
    s_output_one = torch.sigmoid(output_0)
    entropy_loss = F.softmax(s_output_one, dim=2) * \
            F.log_softmax(s_output_one, dim=2)
    entropy_loss = -1 * entropy_loss.sum()
    return entropy_loss

def get_loss_hacky(output_0):
    hacky_loss = -output_0.max()
    return hacky_loss

def get_attention_loss(attention_kind, output_c, target_c):
    if attention_kind == 'hacky':
        loss = get_loss_hacky(output_c)
    elif attention_kind == 'entropy':
        loss = get_loss_entropy(output_c)
    elif attention_kind == 'cheating':
        _, loss, _ = _bce_train_criterion(output_c, target_c)
    else:
        raise NotImplementedError()
    return loss


def forward_hook_gcam(self, input_, output):
    self.gcam_activation = output

def backward_hook_gcam(self, grad_input, grad_output):
    self.gcam_grad = grad_output[0]


def get_attention_gradient_bs1(
        single_gpu_model, input_c, target_c,
        attention_loss_func):

    was_training = single_gpu_model.training
    single_gpu_model.eval()

    assert len(input_c.shape) == 5
    assert len(target_c.shape) == 3

    pass

    gradient = []
    for input0_c, target0_c in zip(input_c, target_c):
        with torch.enable_grad():
            input0_c = input0_c[None].requires_grad_()
            target0_c = target0_c[None]
            output0_c = single_gpu_model(
                    input0_c, None)  # 1, 7, 157
            loss0 = attention_loss_func(output0_c, target0_c)
        gradient0 = torch.autograd.grad(
                    loss0, input0_c,
                    retain_graph=False)
        gradient0 = gradient0[0].detach()  # B, T, H, W, 3
        gradient.append(gradient0)
    gradient = torch.cat(gradient, dim=0)
    if was_training:
        single_gpu_model.train()

    assert len(gradient.shape) == 5

    return gradient


def get_attention_gradcam_bs1(
        single_gpu_model, input_c, target_c,
        attention_loss_func):

    was_training = single_gpu_model.training
    single_gpu_model.eval()

    assert len(input_c.shape) == 5
    assert len(target_c.shape) == 3

    # import pudb; pudb.set_trace()  # XXX BREAKPOINT
    fhook = single_gpu_model.basenet.layer4.register_forward_hook(forward_hook_gcam)
    bhook = single_gpu_model.basenet.layer4.register_full_backward_hook(backward_hook_gcam)

    heatmaps = []
    for input0_c, target0_c in zip(input_c, target_c):
        with torch.enable_grad():
            input0_c_ = input0_c[None]
            target0_c_ = target0_c[None]
            output0_c = single_gpu_model(input0_c_, None)  # 1, 7, 157
            loss0 = attention_loss_func(output0_c, target0_c_)
        loss0.backward(retain_graph=False)
        # GCAM logic
        gcam_activ = single_gpu_model.basenet.layer4.gcam_activation.detach()
        gcam_grad = single_gpu_model.basenet.layer4.gcam_grad
        feature_weights = torch.mean(gcam_grad, dim=(2, 3, 4), keepdim=True)
        weighed_activations = gcam_activ * feature_weights
        # B, T, H_feat, W_feat
        heatmap = torch.functional.F.relu(torch.sum(weighed_activations, dim=1))
        heatmaps.append(heatmap)
    heatmaps = torch.cat(heatmaps, dim=0)
    if was_training:
        single_gpu_model.train()

    # Clean hook
    fhook.remove()
    bhook.remove()

    return heatmaps


def reduce_channel(gradient, kind):
    if kind == 'l2_norm':
        gradient = gradient.norm(dim=-1, keepdim=True)
    elif kind == 'sum':
        gradient = gradient.sum(dim=-1, keepdim=True)
    else:
        raise NotImplementedError()
    return gradient


def reduce_frame(gradient, kind, framedim=0):
    if kind == 'max':
        gradient = gradient.max(dim=framedim, keepdim=True)[0]
    elif kind == 'min':
        gradient = gradient.min(dim=framedim, keepdim=True)[0]
    elif kind == 'mean':
        gradient = gradient.mean(dim=framedim, keepdim=True)
    else:
        raise NotImplementedError()
    return gradient


def get_gradientsum_boxes(
        gradient, _sum_conv, BOX_SIDE):
    assert len(gradient.shape) == 4
    gradient_prm = gradient.permute(0, 3, 1, 2)
    with torch.no_grad():
        box_sum = _sum_conv(gradient_prm)
    assert len(box_sum.shape) == 4 and (box_sum.shape[1] == 1)
    box_sum = box_sum.cpu().numpy()[:, 0]
    ijs = []
    for X in box_sum:
        i, j = np.unravel_index(X.argmax(), X.shape)
        ijs.append((i, j))
    ijs = np.array(ijs)
    boxes = np.c_[ijs, ijs+BOX_SIDE]
    return boxes

def create_boxsum_convolution(BOX_SIDE, in_channels=3):
    # Boxsum convolution
    sum_conv = torch.nn.Conv2d(in_channels, 1, BOX_SIDE, stride=1, bias=False)
    # Set weights to 1
    conv_ones = torch.nn.Parameter(torch.ones_like(
        sum_conv.weight).cuda())
    sum_conv.weight = conv_ones
    return sum_conv


class Networks_wrap(object):
    """
    Wrap around (possibly) multiple torch models + optimizer
     We need this to model our 2net/concat models
    """
    # Attention configs
    channel_reduce = 'l2_norm'
    perframe_reduce = 'max'
    att_crop: int
    att_kind: str

    def __init__(self, model, optimizer, norm_mean, norm_std,
            lr, lr_decay_rate, att_crop, att_kind):
        self.model = model
        self.tmwrap = TModel_wrap(model, norm_mean, norm_std)
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.att_crop = att_crop
        self.att_kind = att_kind
        self.boxsum_conv = create_boxsum_convolution(att_crop, 1)

    def load_my_checkpoint(self, checkpoint_path):
        start_epoch = load_my_checkpoint(
                self.model, self.optimizer, checkpoint_path)
        return start_epoch

    def restore_model_magic(
            self, checkpoint_path, inputs_model,
            training_start_epoch: int, model_type: str):
        """ Complicate strategy to load various checkpoints """
        # If we are continuing from previous checkpoint
        if checkpoint_path:
            start_epoch = self.load_my_checkpoint(checkpoint_path)
            start_epoch += 1
            log.info('Continuing training from checkpoint {}. '
                    'Epoch {} (ckpt + 1)'.format(checkpoint_path, start_epoch))
            return start_epoch

        start_epoch = training_start_epoch
        log.info('Setting start epoch at {}'.format(start_epoch))

        # If model is undefined - start from gunnars model
        if inputs_model is None:
            path = gs_model.get_gunnar_pretrained_path(model_type)
            gs_model.checkpoints_load_simplified(self.model, path)
            log.info('Loaded model from Gunnar')
            return start_epoch

        # If defined - load it
        try:
            # First try loading my model
            self.load_my_checkpoint(inputs_model)
            log.info('Loaded my model from checkpoint {}'.format(inputs_model))
        except KeyError:
            # Then try gunnars loading
            gs_model.checkpoints_load_simplified(self.model, checkpoint_path)
            log.info('Loaded gunnars model from checkpoint {}'.format(inputs_model))
        return start_epoch

    def lr_epoch_adjustment(self, epoch):
        adjust_learning_rate_explicitly(
                self.lr, self.lr_decay_rate, self.optimizer, epoch)

    def checkpoints_save(self, rundir, epoch):
        checkpoints_save(rundir, epoch, self.model, self.optimizer)

    def set_train(self):
        self.model.train()
        self.optimizer.zero_grad()

    def set_eval(self):
        self.model.eval()

    def train_criterion(self, output, target):
        output_ups, loss, target_ups = _bce_train_criterion(
                output, target)
        return output_ups, loss, target_ups

    def eval_criterion(self, output, target):
        output_ups, loss, target_ups = _bce_eval_criterion(
                output, target)
        return output_ups, loss, target_ups

    def _criterions_to_cpudict(self, output_cpu, target_cpu):
        output_ups, loss, target_ups = self.eval_criterion(
                output_cpu, target_cpu)
        _, train_loss, _ = self.train_criterion(
                output_cpu, target_cpu)

        output_ups = output_ups.cpu().numpy()
        loss = loss.cpu().item()
        train_loss = train_loss.cpu().item()
        target_ups = target_ups.cpu().numpy()

        return {'scores': output_ups,
                'loss': loss,
                'train_loss': train_loss,
                'score_target': target_ups}

    def forward_model_for_training(self, X, X_plus, target):
        raise NotImplementedError()

    def forward_model_for_eval_cpu(
            self, X, X_plus, target, batch_size) -> Item_output:
        raise NotImplementedError()

    def get_attention_gradient_v2(
            self, X_u8, target_cpu):
        """
        Attention gradient obtained with self._model network
        """
        X_f32c = self.tmwrap.to_gpu_and_normalize(X_u8)

        # BS=1, prevents crash with dataparallel and instability
        assert isinstance(self.model, MyDataParallel)
        single_gpu_model = self.model.module

        if self.att_kind in ['hacky', 'entropy', 'cheating']:
            def attention_loss_func(output_c, target_c):
                return get_attention_loss(self.att_kind, output_c, target_c)

            gradient = get_attention_gradient_bs1(
                    single_gpu_model, X_f32c,
                    target_cpu.cuda(), attention_loss_func)
            gradient = gradient.abs()
            gradient = reduce_channel(gradient, self.channel_reduce)  # B,T,H,W,1
            gradient = reduce_frame(
                    gradient, self.perframe_reduce, framedim=1)  # B,1,H,W,1
            boxes = get_gradientsum_boxes(
                    gradient[:, 0], self.boxsum_conv, self.att_crop)
        elif self.att_kind in ['entropy_gradcam']:

            def attention_loss_func(output_c, target_c):
                return -1 * get_loss_entropy(output_c)

            # B, T, H_conv, W_conv
            heatmap = get_attention_gradcam_bs1(
                    single_gpu_model, X_f32c,
                    target_cpu.cuda(), attention_loss_func)
            H, W = X_f32c.shape[2:4]
            heatmap = F.interpolate(heatmap, (H, W), mode='bilinear')  # B, T, H, W
            heatmap = torch.max(heatmap, dim=1, keepdim=True)[0]  # B, 1, H, W
            heatmap = torch.unsqueeze(heatmap, dim=4)  # B,1,H,W,1
            boxes = get_gradientsum_boxes(
                    heatmap[:, 0], self.boxsum_conv, self.att_crop)
            gradient = heatmap
        else:
            raise RuntimeError('Unknown attention kind')

        return gradient, boxes


class Networks_wrap_single(Networks_wrap):
    # Single model, everything is simple
    def __init__(self, model, optimizer,
            norm_mean, norm_std, lr, lr_decay_rate, att_crop, att_kind):
        super().__init__(model, optimizer, norm_mean, norm_std,
                lr, lr_decay_rate, att_crop, att_kind)
        self.model = model
        self.tmwrap = TModel_wrap(model, norm_mean, norm_std)
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate

    def forward_model_for_training(self, X, X_plus, target):
        assert X_plus is None
        output_X = self.tmwrap.forward_model(X)
        output_ups, loss, target_ups = self.train_criterion(
                output_X, target.cuda())
        return output_ups, loss, target_ups

    def forward_model_for_eval_cpu(
            self, X, X_plus, target, batch_size) -> Item_output:
        assert X_plus is None
        output_X = self.tmwrap.forward_model_batchwise_nograd(
                X, batch_size).cpu().numpy()
        output_item: Item_output = {
                'output_X': output_X,
                'output_X_plus': None,
                'target': target.numpy()}
        return output_item

    def outputs_items_to_results(
            self, output_items: Dict[str, Item_output]):
        cpudicts = {}
        for vid, output_item in output_items.items():
            cpudicts[vid] = self._criterions_to_cpudict(
                    output_item['output_X'], output_item['target'])
        metrics_charades = _quick_scores_from_cpudicts(cpudicts)
        results: MKinds_Charades = {'normal': metrics_charades}
        return results

class Networks_wrap_stacked(Networks_wrap):
    """
    self.model: <- [X, X_plus]
    """
    def __init__(self, model, optimizer, norm_mean, norm_std,
            lr, lr_decay_rate, att_crop, att_kind):
        super().__init__(model, optimizer, norm_mean, norm_std,
                lr, lr_decay_rate, att_crop, att_kind)

    def forward_model_for_training(self, X, X_plus, target):
        X_concat = torch.cat((X, X_plus), axis=1)
        target_concat = target.repeat(1, 2, 1)
        output_concat = self.tmwrap.forward_model(X_concat)
        output_ups, loss, target_ups = self.train_criterion(
                output_concat, target_concat.cuda())
        return output_ups, loss, target_ups

    def forward_model_for_eval_cpu(
            self, X, X_plus, target, batch_size) -> Item_output:
        X_concat = torch.cat((X, X_plus), axis=1)  # type: ignore
        output_concat = self.tmwrap.forward_model_batchwise_nograd(
                X_concat, batch_size).cpu().numpy()
        target_concat = target.repeat(1, 2, 1).numpy()
        output_item: Item_output = {
                'output_X': output_concat,
                'output_X_plus': None,
                'target': target_concat}
        return output_item

    def outputs_items_to_results(
            self, output_items: Dict[str, Item_output]):
        cpudicts = {}
        for vid, output_item in output_items.items():
            cpudicts[vid] = self._criterions_to_cpudict(
                    output_item['output_X'], output_item['target'])
        metrics_charades = _quick_scores_from_cpudicts(cpudicts)
        results: MKinds_Charades = {'concat': metrics_charades}
        return results

class Networks_wrap_twonet(Networks_wrap):
    """
    self.model_fixed: <- X
    self.model: <- X_plus
    """
    def __init__(self, model, optimizer, norm_mean, norm_std,
            lr, lr_decay_rate, att_crop, att_kind, fixed_net_path):
        super().__init__(model, optimizer, norm_mean, norm_std,
                lr, lr_decay_rate, att_crop, att_kind)
        self.fixed_net_path = fixed_net_path
        self.model_fixed = copy.deepcopy(self.model)
        self.tmwrap_fixed = TModel_wrap(self.model_fixed, norm_mean, norm_std)
        # Load fixed model right away
        states = torch.load(self.fixed_net_path)
        self.model_fixed.load_state_dict(states['model_sdict'])

    def forward_model_for_training(self, X, X_plus, target):
        with torch.no_grad():
            output_X = self.tmwrap_fixed.forward_model(X)
        output_X_plus = self.tmwrap.forward_model(X_plus)
        stacked_outputs = torch.stack((output_X, output_X_plus))
        reduced = torch.mean(stacked_outputs, dim=0)
        output_ups, loss, target_ups = self.train_criterion(
                reduced, target.cuda())
        return output_ups, loss, target_ups

    def forward_model_for_eval_cpu(
            self, X, X_plus, target, batch_size) -> Item_output:
        output_X = self.tmwrap_fixed.forward_model_batchwise_nograd(
                X, batch_size).cpu().numpy()
        output_X_plus = self.tmwrap.forward_model_batchwise_nograd(
                X_plus, batch_size).cpu().numpy()
        output_item: Item_output = {
                'output_X': output_X,
                'output_X_plus': output_X_plus,
                'target': target.numpy()}
        return output_item

    def outputs_items_to_results(
            self, output_items: Dict[str, Item_output]):
        cpudicts_X = {}
        cpudicts_X_plus = {}
        cpudicts_reduced = {}
        for vid, output_item in output_items.items():
            assert output_item['output_X_plus'] is not None
            stacked_outputs = torch.stack((
                output_item['output_X'], output_item['output_X_plus']))
            reduced = torch.mean(stacked_outputs, dim=0)
            cpudicts_X[vid] = self._criterions_to_cpudict(
                    output_item['output_X'], output_item['target'])
            cpudicts_X_plus[vid] = self._criterions_to_cpudict(
                    output_item['output_X_plus'], output_item['target'])
            cpudicts_reduced[vid] = self._criterions_to_cpudict(
                    reduced, output_item['target'])
        metrics_X = _quick_scores_from_cpudicts(cpudicts_X)
        metrics_X_plus = _quick_scores_from_cpudicts(cpudicts_X_plus)
        metrics_reduced = _quick_scores_from_cpudicts(cpudicts_reduced)
        results: MKinds_Charades = {
                'normal': metrics_X,
                'zoomed': metrics_X_plus,
                'reduced': metrics_reduced}
        return results

    def set_train(self):
        super().set_train()
        self.model_fixed.train()

    def set_eval(self):
        super().set_eval()
        self.model_fixed.eval()
