import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

import numpy as np

from utils.builder import get_builder
from args import args

builder = get_builder()

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = builder.batchnorm(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = builder.conv3x3(in_planes, out_planes, stride=stride)
        self.bn2 = builder.batchnorm(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = builder.conv3x3(out_planes, out_planes, stride=1)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and builder.conv1x1(in_planes, out_planes, stride=stride) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = builder.conv3x3(3, nChannels[0], stride=1)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = builder.batchnorm(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        if args.last_layer_dense:
            self.fc = nn.Conv2d(nChannels[3], num_classes, 1)
        else:
            self.fc = builder.conv1x1(nChannels[3], num_classes)

        self.nChannels = nChannels[3]

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)

        out = self.fc(out)
        return out.flatten(1)

        # out = out.view(-1, self.nChannels)
        # return self.fc(out)


    # def set_precision(self, num_bits=None, num_grad_bits=None):
    #     for module in self.modules():
    #         if isinstance(module, QConv2d):
    #             module.set_precision(num_bits, num_grad_bits)
    #         if isinstance(module, USBatchNorm2d):
    #             module.set_precision(num_bits)


    # def set_stochastic(self, stochastic=False):
    #     for module in self.modules():
    #         if isinstance(module, QConv2d):
    #             module.set_stochastic(stochastic)


    # def set_random_quant(self, num_bits=None, num_grad_bits=None, random_quant_prob=0.5):
    #     for module in self.modules():
    #         if isinstance(module, QConv2d):
    #             flag_quant = np.random.choice([True, False], p=[random_quant_prob, 1-random_quant_prob])

    #             if flag_quant:
    #                 module_num_bits = num_bits
    #                 module_num_grad_bits = num_grad_bits
    #             else:
    #                 module_num_bits = 0
    #                 module_num_grad_bits = 0

    #             module.set_precision(module_num_bits, module_num_grad_bits)


def WideResNet32(num_classes=10):
    return WideResNet(num_classes=num_classes)