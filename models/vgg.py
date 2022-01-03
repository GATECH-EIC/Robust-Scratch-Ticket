import torch
import torch.nn as nn
from utils.builder import get_builder
from args import args


builder = get_builder()


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

        if args.last_layer_dense:
            self.classifier = nn.Conv2d(512, num_classes, 1)
        else:
            self.classifier = builder.conv1x1(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        # out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out.flatten(1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [builder.conv3x3(in_channels, x),
                           builder.batchnorm(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(num_classes=10):
    return VGG('VGG11', num_classes=num_classes)


def VGG16(num_classes=10):
    return VGG('VGG16', num_classes=num_classes)