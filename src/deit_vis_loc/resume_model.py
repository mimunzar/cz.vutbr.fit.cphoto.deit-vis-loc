#!/usr/bin/env python3

import argparse
import sys
import os
from time import time
from datetime import datetime

import torch

import src.deit_vis_loc.train_model as train_model
import src.deit_vis_loc.libs.util as util


def make_outdirs(model_path, dataset, **_):
    model_path = os.path.expanduser(model_path)
    output_dir = util.first(os.path.split(os.path.dirname(model_path)))
    timestr    = datetime.fromtimestamp(time()).strftime('%Y%m%dT%H%M%S')
    result_dir = os.path.join(output_dir, f'{timestr}-{dataset}')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def load_net(model_path, **_):
    return torch.load(model_path)


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',  help='The name of the model',
            required=True, metavar='FILE')
    parser.add_argument('--input-size',  help='The resolution of input images',
            required=True, type=int, metavar='INT')
    parser.add_argument('--data-dir',    help='The path to the dataset',
            required=True, metavar='DIR')
    parser.add_argument('--dataset',     help='The type of the input dataset',
            required=True, choices=['sparse', 'pretraining'])
    parser.add_argument('--modality',    help='The modality of images',
            required=True, choices=['segments', 'silhouettes', 'depth'])
    parser.add_argument('--n-images',    help='The number of images',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--params',      help='The file path to training definition',
            required=True, metavar='FILE')
    parser.add_argument('--device',      help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--gpu-imcap',   help='The amount of images to fit on GPU',
            required=True, type=int, metavar='INT')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    params = train_model.parse_params(**args)
    train_model.start(make_outdirs(**args), load_net(**args), args, params)
    sys.exit(0)

