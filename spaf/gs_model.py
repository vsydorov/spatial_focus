import logging
import os
import collections
import torch.nn as nn
from collections import OrderedDict

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(
                inplanes, planes, kernel_size=(3, 1, 1),
                padding=(1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv3d(
                planes, planes, kernel_size=(1, 3, 3),
                stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv3d(
                planes, planes * self.expansion,
                kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(
                planes * self.expansion, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self, block, layers, num_classes=400):
        super(ResNet3D, self).__init__()
        self.global_pooling = False
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
                3, 64, kernel_size=(5, 7, 7), stride=2,
                padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
                kernel_size=3, stride=2, padding=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.maxpool2 = nn.MaxPool3d(
                kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(.5)
        self.fc = nn.Conv3d(
                512 * block.expansion, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x is of the form b x n x h x w x c
        # model expects b x c x n x h x w
        x = x.permute(0, 4, 1, 2, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        logits = self.fc(x)

        logits = logits.mean(3).mean(3)
        # model returns batch x classes x time
        logits = logits.permute(0, 2, 1)
        # logits is batch X time X classes
        if self.global_pooling:
            logits = logits.mean(1)
        return logits

    def load_2d(self, model2d):
        log.info('inflating 2d resnet parameters')
        sd = self.state_dict()
        sd2d = model2d.state_dict()
        sd = OrderedDict(
                [(x.replace('module.', ''), y) for x, y in sd.items()])
        sd2d = OrderedDict(
                [(x.replace('module.', ''), y) for x, y in sd2d.items()])
        for k, v in sd2d.items():
            if k not in sd:
                log.info('ignoring state key for loading: {}'.format(k))
                continue
            if 'conv' in k or 'downsample.0' in k:
                s = sd[k].shape
                t = s[2]
                sd[k].copy_(v.unsqueeze(2).expand(*s) / t)
            elif 'bn' in k or 'downsample.1' in k:
                sd[k].copy_(v)
            else:
                log.info('skipping: {}'.format(k))

    def replace_logits(self, num_classes):
        self.fc = nn.Conv3d(self.fc.in_channels, num_classes, kernel_size=1)


class _NonLocalBlockND(nn.Module):
    # @AlexHex7
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True, group_size=None, zero_init_conv=False):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.group_size = group_size

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.g.weight, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            if zero_init_conv:
                nn.init.constant_(self.W[1].weight, 0)
            else:
                nn.init.normal_(self.W[1].weight, std=0.01)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            if zero_init_conv:
                nn.init.constant_(self.W.weight, 0)
            else:
                nn.init.normal_(self.W.weight, std=0.01)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.normal_(self.theta.weight, std=0.01)
            nn.init.constant_(self.theta.bias, 0)
            nn.init.normal_(self.phi.weight, std=0.01)
            nn.init.constant_(self.phi.bias, 0)

            if mode == 'concatenation':
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        if self.group_size is not None:
            b, c, t, h, w = x.shape
            # x = x.reshape(-1, c, self.group_size, h, w)
            x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape(-1, self.group_size, c, h, w)
            x = x.permute(0, 2, 1, 3, 4)

        if self.mode == 'embedded_gaussian':
            output = self._embedded_gaussian(x)
        elif self.mode == 'dot_product':
            output = self._dot_product(x)
        elif self.mode == 'concatenation':
            output = self._concatenation(x)
        elif self.mode == 'gaussian':
            output = self._gaussian(x)

        if self.group_size is not None:
            b2, c2, t2, h2, w2 = output.shape
            output = output.permute(0, 2, 1, 3, 4)
            output = output.reshape(b, -1, c2, h2, w2)
            output = output.permute(0, 2, 1, 3, 4)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,
            mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,
            mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None,
            mode='embedded_gaussian', sub_sample=True, bn_layer=True, group_size=None):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              group_size=group_size)


class ResNet3DNonLocal(ResNet3D):
    def __init__(self, block, layers, num_classes=400):
        super(ResNet3DNonLocal, self).__init__(block, layers, num_classes)

    def insert_nonlocal_blocks(self, nonlocal_blocks):
        for layername, nr in zip(['layer1', 'layer2', 'layer3', 'layer4'], nonlocal_blocks):
            if nr == 0:
                continue
            layers = getattr(self, layername)
            newlayers = []
            insert_freq = len(layers) / nr
            for i, layer in enumerate(layers):
                newlayers.append(layer)
                if i % insert_freq == 0:
                    n = layer.conv3.out_channels
                    if layername == 'layer2':
                        blocknl = NONLocalBlock3D(n, group_size=4)
                    else:
                        blocknl = NONLocalBlock3D(n)
                    newlayers.append(blocknl)

            newlayers = nn.Sequential(*newlayers)
            setattr(self, layername, newlayers)


def split_list(lst, chunk_num):
    n = len(lst)
    chunk_size = max(1, n // chunk_num)
    i = 0
    out_lst = []
    while i < n:
        out_lst.append(lst[i:i+chunk_size])
        i += chunk_size
    return tuple(out_lst)


class MyDataParallel(nn.DataParallel):
    # Overloads nn.DataParallel to provide the ability to skip
    # scatter/gather functionality for a simple unmodified
    # list of dictionaries
    def __init__(self, *args, **kwargs):
        super(MyDataParallel, self).__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        # only scatter inputs that don't have the do_not_collate flag
        inputss = []
        for inp in inputs:
            if isinstance(inp, collections.abc.Sequence):
                if isinstance(inp[0], collections.abc.Mapping) and \
                        'do_not_collate' in inp[0]:
                    inp = split_list(inp, len(device_ids))
                else:
                    inp, kwargs = super(MyDataParallel, self).scatter(
                            (inp, ), kwargs, device_ids)
                    inp = [x[0] if x != () else x for x in inp]  # de-tuple
            else:
                inp, kwargs = super(MyDataParallel, self).scatter(
                        (inp, ), kwargs, device_ids)
                inp = [x[0] if x != () else x for x in inp]  # de-tuple
                if isinstance(kwargs[0], collections.abc.Sequence):
                    kwargs = [x[0] if x != () else x
                            for x in kwargs]  # de-tuple
            inputss.append(inp)
        return tuple(zip(*inputss)), kwargs

    def gather(self, outputs, output_device):
        # only gather outputs that don't have the do_not_collate flag
        # return should be #args x #batch
        outputss = []
        if isinstance(outputs[0], tuple):
            # multiple output arguments
            # outputs is #gpu x #args x #gpu_batch
            for out in zip(*outputs):  # out is #gpu x #gpu_batch
                if (isinstance(out[0], collections.abc.Sequence) and
                   isinstance(out[0][0], collections.abc.Mapping) and
                   'do_not_collate' in out[0][0]):
                    out = [x for y in out for x in y]  # join lists
                else:
                    out = super(MyDataParallel, self).gather(
                            out, output_device)
                outputss.append(out)
            return tuple(outputss)
        else:
            # outputs is #gpu x #gpu_batch
            return super(MyDataParallel, self).gather(outputs, output_device)


def ordered_load_state(model, chkpoint):
    """
        Wrapping the model with parallel/dataparallel seems to
        change the variable names for the states
        This attempts to load normally and otherwise aligns the labels
        of the two states and tries again.
    """
    try:
        model.load_state_dict(chkpoint)
    except RuntimeError as e:  # assume order is the same, and use new labels
        log.debug('Fail to load ordered', exc_info=e)
        log.info('keys do not match model, trying to align')
        model_keys = model.state_dict().keys()
        fixed = OrderedDict([(z, y) for (_, y), z in zip(
            chkpoint.items(), model_keys)])
        model.load_state_dict(fixed)


def load_partial_state(model, state_dict):
    # @chenyuntc
    sd = model.state_dict()
    sd = OrderedDict([(x.replace('module.', '')
        .replace('mA.', '')
        .replace('basenet.', '')
        .replace('encoder.', ''), y) for x, y in sd.items()])
    for k0, v in state_dict.items():
        k = k0.replace('module.', '').replace('mA.', '').replace(
                'basenet.', '').replace('encoder.', '')
        if k not in sd or not sd[k].shape == v.shape:
            log.info('ignoring state key for loading: {}'.format(k))
            continue
        if isinstance(v, torch.nn.Parameter):
            v = v.data
        sd[k].copy_(v)


def checkpoints_load(args, model, optimizer):
    for resume in args.resume.split(';'):
        if os.path.isfile(resume):
            log.info("=> loading checkpoint '{}'".format(resume))
            chkpoint = torch.load(resume)
            if isinstance(chkpoint, dict) and 'state_dict' in chkpoint:
                try:
                    ordered_load_state(model, chkpoint['state_dict'])
                    optimizer.load_state_dict(chkpoint['optimizer'])
                    log.info('Loaded optimizer state')
                except Exception as e:
                    log.exception(e)
                    log.info('loading partial state 2')
                    load_partial_state(model, chkpoint['state_dict'])
                log.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume, chkpoint['epoch']))
                if args.start_epoch == 0:
                    args.start_epoch = chkpoint['epoch']
                    log.info('setting start epoch to model epoch {}'.format(
                        args.start_epoch))
                if 'scores' in chkpoint and args.metric in chkpoint['scores']:
                    best_metric = chkpoint['scores'][args.metric]
                else:
                    best_metric = 0
                return best_metric
            else:
                try:
                    ordered_load_state(model, chkpoint)
                except Exception as e:
                    log.exception(e)
                    log.info('loading partial state')
                    load_partial_state(model, chkpoint)
                log.info("=> loaded checkpoint '{}' (just weights)".format(
                    resume))
                return 0
            break
        else:
            log.info(("=> no checkpoint found, "
                    "starting from scratch: '{}'").format(resume))
    return 0


def checkpoints_load_simplified(model, checkpoint):
    chkpoint = torch.load(checkpoint)
    try:
        ordered_load_state(model, chkpoint['state_dict'])
    except Exception as e:
        log.info('Fail to load full state', exc_info=e)
        log.info('loading partial state 2')
        load_partial_state(model, chkpoint['state_dict'])
    log.info("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint, chkpoint['epoch']))


def get_gunnar_pretrained_path(model_type):
    if model_type == 'resnet50':
        path = '/home/vsydorov/projects/deployed/2019_01_ICCV_video_attention/links/gpuhost7/dev_area/10_2019_01_gunnar/models/i3d8k.pth.tar'
    elif model_type == 'nl-resnet50':
        path = '/home/vsydorov/projects/deployed/2019_01_ICCV_video_attention/links/gpuhost7/dev_area/10_2019_01_gunnar/models/i3d8l.pth.tar'
    else:
        raise NotImplementedError()
    return path


class ExplicitWrapper(torch.nn.Module):
    def __init__(self, basenet, freeze_batchnorm):
        super().__init__()
        self.basenet = basenet
        self.freeze_batchnorm = freeze_batchnorm

    def forward(self, x, meta):
        if self.freeze_batchnorm:
            for module in self.basenet.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
        return self.basenet(x)


def explicit_replace_last_layer(model, nclass):
    if hasattr(model, 'replace_logits'):
        model.replace_logits(nclass)
    elif hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        model.classifier = torch.nn.Sequential(*newcls[:-1])
    elif hasattr(model, 'fc'):
        model.fc = torch.nn.Linear(model.fc.in_features, nclass)
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = torch.nn.Linear(
                    model.AuxLogists.fc.in_features, nclass)
    else:
        newcls = list(model.children())[:-1]
        model = torch.nn.Sequential(*newcls)
    return model


def get_explicit_model_i3d(nclass):
    # Model
    pretrained = True
    replace_last_layer = True
    freeze_batchnorm = False
    model = ResNet3D(Bottleneck3D, [3, 4, 6, 3])  # 50
    if pretrained:
        from torchvision.models.resnet import resnet50
        model2d = resnet50(pretrained=True)
        model.load_2d(model2d)
    if replace_last_layer:
        model = explicit_replace_last_layer(model, nclass)
    # Wrapper
    model = ExplicitWrapper(model, freeze_batchnorm=freeze_batchnorm)
    # "set_distributed_backend" when distributed=False
    if hasattr(model, 'features'):
        model.features = MyDataParallel(model.features)
        model.cuda()
    else:
        model = MyDataParallel(model).cuda()
    return model


def get_explicit_model_i3d_nonlocal(nclass):
    # Model
    pretrained = True
    replace_last_layer = True
    freeze_batchnorm = False
    model = ResNet3DNonLocal(Bottleneck3D, [3, 4, 6, 3])  # 50
    if pretrained:
        from torchvision.models.resnet import resnet50
        model2d = resnet50(pretrained=True)
        model.load_2d(model2d)
    model.insert_nonlocal_blocks([0, 2, 3, 0])
    if replace_last_layer:
        model = explicit_replace_last_layer(model, nclass)
    # Wrapper
    model = ExplicitWrapper(model, freeze_batchnorm=freeze_batchnorm)
    # "set_distributed_backend" when distributed=False
    if hasattr(model, 'features'):
        model.features = MyDataParallel(model.features)
        model.cuda()
    else:
        model = MyDataParallel(model).cuda()
    return model
