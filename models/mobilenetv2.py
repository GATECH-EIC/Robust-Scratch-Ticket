import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

import numpy as np

from utils.builder import get_builder
from args import args

builder = get_builder()


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = builder.conv1x1(in_planes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, groups=planes)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, out_planes)
        self.bn3 = builder.batchnorm(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, out_planes),
                builder.batchnorm(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_Net(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_Net, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = builder.conv3x3(3, 32, stride=1)
        self.bn1 = builder.batchnorm(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = builder.conv1x1(320, 1280)
        self.bn2 = builder.batchnorm(1280)

        if args.last_layer_dense:
            self.fc = nn.Conv2d(1280, num_classes, 1)
        else:
            self.fc = builder.conv1x1(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)

        out = self.fc(out)
        return out.flatten(1)
    
    # def set_precision(self, num_bits=None, num_grad_bits=None):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             module.set_precision(num_bits, num_grad_bits)


    # def set_stochastic(self, stochastic=False):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             module.set_stochastic(stochastic)


    # def set_random_quant(self, num_bits=None, num_grad_bits=None, random_quant_prob=0.5):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             flag_quant = np.random.choice([True, False], p=[random_quant_prob, 1-random_quant_prob])

    #             if flag_quant:
    #                 module_num_bits = num_bits
    #                 module_num_grad_bits = num_grad_bits
    #             else:
    #                 module_num_bits = 0
    #                 module_num_grad_bits = 0

    #             module.set_precision(module_num_bits, module_num_grad_bits)


def MobileNetV2(num_classes=10):
    return MobileNetV2_Net(num_classes=num_classes)