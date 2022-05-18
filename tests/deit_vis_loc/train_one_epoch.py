#!/usr/bin/env python3

import argparse
import functools as ft
import random
import sys

import torch.optim

import tests.deit_vis_loc.commons as commons
import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.locate as locate
import src.deit_vis_loc.training.model as model
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   help='The path to dataset',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',   help='The amount of images to sample from the Dataset',
            required=True, type=int, metavar='INT')
    parser.add_argument('--resolution', help='The resolution of output images',
            required=False, metavar='INT', type=int, default=224)
    parser.add_argument('--device',     help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args          = parse_args(sys.argv[1:])
    data_dir, res = util.pluck(['data_dir', 'resolution'], args)
    n_im, device  = util.pluck(['n_images', 'device'],     args)
    params        = {
        'deit_model'       : 'deit_tiny_patch16_224',
        'input_size'       : res,
        'margin'           : 0.1,
        'lr'               : 1e-3,
        'batch_size'       : 5,
        'mine_every_epoch' : 1,
        **commons.PRETRAINING_PARAMS,
    }

    net      = model.load(params['deit_model']).to(device)
    optim    = torch.optim.SGD(net.parameters(), params['lr'], momentum=0.9)
    fwd      = util.compose(net, ft.partial(locate.load_im, device))
    im_tp_it = locate.iter_im_triplet(device, params, fwd,
            random.sample(tuple(loader.iter_queries(data_dir, res, 'train')), k=args['n_images']),
            loader.iter_pretraining_renders        (data_dir, res, 'segments'))
    result   = locate.epochstat({
            'device': 'cuda',
            'net'   : net,
            'optim' : optim,
        }, params, 1, im_tp_it)

