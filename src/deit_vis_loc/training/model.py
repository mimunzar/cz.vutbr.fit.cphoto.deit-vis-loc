#!/usr/bin/env python3

import torch.hub
import torch.nn


def load(deit_name):
    net      = torch.hub.load('facebookresearch/deit:main', deit_name, pretrained=True)
    net.norm = torch.nn.Identity()
    net.head = torch.nn.Identity()
    return net

