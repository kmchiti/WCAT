"""
Modified from https://github.com/chenyaofo/pytorch-cifar-models
"""
from .base_model import Base_Model
import sys
import torch.nn as nn
import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional

cifar10_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
}

cifar100_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
}

__all__ = [
    "Resnet20",
    "Resnet32",
    "Resnet44",
    "Resnet56"
]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=None, num_classes=10):
        super(CifarResNet, self).__init__()
        if layers is None:
            layers = [3, 3, 3]
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

@Base_Model.register('resnet20')
class Resnet20(Base_Model, CifarResNet):
    def __init__(self, num_classes=10, pretrained=False, progress=True,
                 **kwargs):
        CifarResNet.__init__(self, block=BasicBlock, layers=[3, 3, 3], num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)
        if pretrained:
            if num_classes == 10:
                _model_url = cifar10_pretrained_weight_urls['resnet20']
            elif num_classes == 100:
                _model_url = cifar100_pretrained_weight_urls['resnet20']
            else:
                raise f"Not found pretrained model for num_classes {num_classes}"
            state_dict = load_state_dict_from_url(_model_url, self.save_path,
                                                  map_location=self.device, progress=progress)
            network_kvpair = self.state_dict()
            for key in state_dict.keys():
                network_kvpair[key] = state_dict[key]
            self.load_state_dict(network_kvpair)

@Base_Model.register('resnet32')
class Resnet32(CifarResNet, Base_Model):
    def __init__(self, num_classes=10, pretrained=False, progress=True,
                 **kwargs):
        CifarResNet.__init__(self, block=BasicBlock, layers=[5, 5, 5], num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)
        if pretrained:
            if num_classes == 10:
                _model_url = cifar10_pretrained_weight_urls['resnet32']
            elif num_classes == 100:
                _model_url = cifar100_pretrained_weight_urls['resnet32']
            else:
                raise f"Not found pretrained model for num_classes {num_classes}"
            state_dict = load_state_dict_from_url(_model_url, self.save_path,
                                                  map_location=self.device, progress=progress)
            network_kvpair = self.state_dict()
            for key in state_dict.keys():
                network_kvpair[key] = state_dict[key]
            self.load_state_dict(network_kvpair)

@Base_Model.register('resnet44')
class Resnet44(CifarResNet, Base_Model):
    def __init__(self, num_classes=10, pretrained=False, progress=True,
                 **kwargs):
        CifarResNet.__init__(self, block=BasicBlock, layers=[7, 7, 7], num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)
        if pretrained:
            if num_classes == 10:
                _model_url = cifar10_pretrained_weight_urls['resnet44']
            elif num_classes == 100:
                _model_url = cifar100_pretrained_weight_urls['resnet44']
            else:
                raise f"Not found pretrained model for num_classes {num_classes}"
            state_dict = load_state_dict_from_url(_model_url, self.save_path,
                                                  map_location=self.device, progress=progress)
            network_kvpair = self.state_dict()
            for key in state_dict.keys():
                network_kvpair[key] = state_dict[key]
            self.load_state_dict(network_kvpair)

@Base_Model.register('resnet56')
class Resnet56(CifarResNet, Base_Model):
    def __init__(self, num_classes=10, pretrained=False, progress=True,
                 **kwargs):
        CifarResNet.__init__(self, block=BasicBlock, layers=[9, 9, 9], num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)

        if pretrained:
            if num_classes == 10:
                _model_url = cifar10_pretrained_weight_urls['resnet56']
            elif num_classes == 100:
                _model_url = cifar100_pretrained_weight_urls['resnet56']
            else:
                raise f"Not found pretrained model for num_classes {num_classes}"
            state_dict = load_state_dict_from_url(_model_url, self.save_path,
                                                  map_location=self.device, progress=progress)
            network_kvpair = self.state_dict()
            for key in state_dict.keys():
                network_kvpair[key] = state_dict[key]
            self.load_state_dict(network_kvpair)
