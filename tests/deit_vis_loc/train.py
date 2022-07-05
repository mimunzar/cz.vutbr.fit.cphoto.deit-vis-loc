#!/usr/bin/env python3

import argparse
import sys

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.crosslocate as crosslocate
import src.deit_vis_loc.training.zoo as zoo
from tests.deit_vis_loc.params import PRETRAINING_PARAMS


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='The model identifier',
            required=True, choices=[
                'deit_base_patch16_224',
                'deit_base_patch16_384',
                'deit_small_patch16_224',
                'deit_tiny_patch16_224'])
    parser.add_argument('--input-size',  help='The resolution of input images',
            required=True, type=int, metavar='INT')
    parser.add_argument('--data-dir', help='The path to the dataset',
            required=True, metavar='DIR')
    parser.add_argument('--dataset', help='The type of the input dataset',
            required=True, choices=['sparse', 'pretraining'])
    parser.add_argument('--modality', help='The modality of images',
            required=True, choices=['segments', 'silhouettes', 'depth'])
    parser.add_argument('--scale-by-fov', help='When set scales images by their FOV',
            required=False, action="store_true")
    parser.add_argument('--n-images', help='The number of images',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--device', help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--gpu-imcap', help='The amount of images to fit on GPU',
            required=True, type=int, metavar='INT')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args     = parse_args(sys.argv[1:])
    device   = args['device']
    n_images = args['n_images']

    params = {
        **PRETRAINING_PARAMS,
        'max_epochs'       : 100,
        'margin'           : 0.1,
        'lr'               : 1e-3,
        'min_lr'           : 1e-5,
        'batch_size'       : 5,
        'mine_every_epoch' : 2,
    }
    net   = zoo.new(args['model_name']).to(device)
    optim = SGD(net.parameters(), params['lr'], momentum=0.9)
    model = {
        'device'    : device,
        'gpu_imcap' : args['gpu_imcap'],
        'net'       : net,
        'optim'     : optim,
        'scheduler' : CosineAnnealingLR(optim, *util.pluck(['max_epochs', 'min_lr'], params)),
    }
    result = crosslocate.iter_trainingepoch(model, params,
            util.take(n_images, loader.iter_queries('val',   **args)),
            util.take(n_images, loader.iter_queries('train', **args)),
            loader.iter_renders_pretraining(**args))

