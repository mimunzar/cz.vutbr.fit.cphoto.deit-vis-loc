#!/usr/bin/env python3

import torch.hub
import torch.nn


def load(deit_name):
    net      = torch.hub.load('facebookresearch/deit:main', deit_name, pretrained=True)
    net.head = torch.nn.Identity()
    return net

