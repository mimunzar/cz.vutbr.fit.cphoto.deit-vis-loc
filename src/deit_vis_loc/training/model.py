#!/usr/bin/env python3

import torch.hub


def load(deit_name):
    return torch.hub.load('facebookresearch/deit:main',
            deit_name, pretrained=True, verbose=False)

