'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from models.layers import *

__all__ = [
    'VGG', 'vgg11', 'vgg13', 'vgg16',
]


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, cfg, num_classes=10, batch_norm=True, in_c=3, **lif_parameters):
        super(VGG, self).__init__()

        self.features, out_c = make_layers(cfg, batch_norm, in_c, **lif_parameters)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Linear(out_c, num_classes)),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.add_dim = lambda x: add_dimention(x, self.T)

    def forward(self, x):
        x = self.add_dim(x) if len(x.shape) == 4 else x
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, in_c=3, **lif_parameters):
    layers = []
    in_channels = in_c
    for v in cfg:
        if v == 'M':
            layers += [SpikeModule(nn.AvgPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = SpikeModule(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))

            lif = LIFSpike(**lif_parameters)

            if batch_norm:
                bn = tdBatchNorm(v)
                layers += [conv2d, bn, lif]
            else:
                layers += [conv2d, lif]

            in_channels = v
    return nn.Sequential(*layers), in_channels


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],
}


def vgg11(*args, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(cfg['A'], *args, **kwargs)


def vgg13(*args, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(cfg['B'], *args, **kwargs)


def vgg16(*args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['D'], *args, **kwargs)


if __name__ == '__main__':
    model = vgg16(num_classes=10, width_mult=1)
    print(model)
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    y.sum().backward()
