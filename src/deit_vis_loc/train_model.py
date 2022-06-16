#!/usr/bin/env python3

import argparse
import json
import os
import sys
from time import time
from datetime import datetime

import torch
import torch.cuda
import torch.hub
import torch.optim

import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.config as config
import src.deit_vis_loc.training.crosslocate as crosslocate
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util


def start(out_dir, net, args, params):
    print(log.msg(f'Copying params to {out_dir}'))
    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def device_name(device, **_):
        return torch.cuda.get_device_name(device) if 'cuda' == device else 'CPU'

    def build_model(net, lr, device, gpu_imcap, **_):
        net.to(device)
        return {
            'net'       : net,
            'device'    : device,
            'gpu_imcap' : gpu_imcap,
            'optim'     : torch.optim.SGD(net.parameters(), lr, momentum=0.9),
        }

    def load_data(n_images, data_dir, dataset, input_size, modality, **_):
        render_loaders = {
            'pretraining' : loader.iter_pretraining_renders,
            'sparse'      : loader.iter_sparse_renders,
        }
        return (
            util.take(n_images, loader.iter_queries(data_dir, input_size, 'val')),
            util.take(n_images, loader.iter_queries(data_dir, input_size, 'train')),
            render_loaders[dataset](data_dir, input_size, modality))

    print(log.msg(
        f'Starting training process "{os.getpid()}" on "{device_name(**args)}"\n'))
    crosslocate.train(build_model(net, **args, **params),
            params, out_dir, *load_data(**args, **params))
    print(log.msg('Finished training', prefix='\n'))


def load_net(model_name, **_):
    return torch.hub.load('facebookresearch/deit:main',
            model_name, pretrained=True, verbose=False)


def make_outdirs(output_dir, dataset, model_name, **_):
    timestr    = datetime.fromtimestamp(time()).strftime('%Y%m%dT%H%M%S')
    result_dir = os.path.join(
        os.path.expanduser(output_dir),
        f'{timestr}-{model_name}',
        f'{timestr}-{dataset}',
    )
    os.makedirs(result_dir, exist_ok=True)
    return result_dir
    #^ Because a model is trained in multiple stages on different datasets,
    # there are two nested directories created. The outer directory contains
    # all training runs of the model. The inned directory contains output from
    # a specific run on a given dataset.


def parse_params(params, **_):
    with open(params, 'r') as f:
        return config.parse(json.load(f))


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',  help='The path to saved model',
            required=True, choices=[
                'deit_base_patch16_224',
                'deit_base_patch16_384',
                'deit_small_patch16_224',
                'deit_tiny_patch16_224'])
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
    parser.add_argument('--output-dir',  help='The output directory',
            required=True, metavar='DIR')
    parser.add_argument('--params',      help='The file path to training definition',
            required=True, metavar='FILE')
    parser.add_argument('--device',      help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--gpu-imcap',   help='The amount of images to fit to GPU',
            required=True, type=int, metavar='INT')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args    = parse_args(sys.argv[1:])
    params  = parse_params(**args)
    start(make_outdirs(**args, **params), load_net(**args), args, params)
    sys.exit(0)

