"""Defines the neural network"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import deeplabv3_resnet50

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['none', nn.Identity()]
    ])[activation]

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

    def forward(self, s):
        pass

def get_network(params):
    if params.architecture == 'deeplab_resnet101':
        net = deeplabv3_resnet101(pretrained=params.pretrained)
    if params.architecture == 'deeplab_resnet50':
        net = deeplabv3_resnet50(pretrained=params.pretrained)
    if 'activation' in params.dict:
        net.classifier[-1] = nn.Sequential(nn.Conv2d(256, params.num_classes, 1, 1 ), activation_func(params.activation))
    return net
    