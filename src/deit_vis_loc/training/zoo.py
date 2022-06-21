#!/usr/bin/env python3

import torch.hub


def new(model_name):
    return torch.hub.load('facebookresearch/deit:main',
            model_name, pretrained=True, verbose=False)

