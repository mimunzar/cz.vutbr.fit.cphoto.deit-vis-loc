#!/usr/bin/env python3

import torch
import torch.hub


def new(deit_name):
    return torch.hub.load('facebookresearch/deit:main',
            deit_name, pretrained=True, verbose=False)


def load(fpath):
    return torch.load(fpath)

