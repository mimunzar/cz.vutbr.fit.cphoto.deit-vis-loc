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
    parser.add_argument('--n-images',   help='The amount of images to sample',
            required=True, type=int, metavar='INT')
    parser.add_argument('--device',     help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    params = {
        'deit_model'       : 'deit_tiny_patch16_224',
        'input_size'       : 224,
        'margin'           : 0.1,
        'lr'               : 1e-3,
        'batch_size'       : 5,
        'mine_every_epoch' : 1,
        **commons.PRETRAINING_PARAMS,
    }

    net    = model.load(params['deit_model']).to(args['device'])
    optim  = torch.optim.SGD(net.parameters(), params['lr'], momentum=0.9)
    fwd    = util.compose(net, ft.partial(locate.load_im, args['device']))
    result = locate.epochstat(optim, params, fwd, 1, locate.iter_im_triplet(
        args['device'], params, fwd,
        random.sample(tuple(loader.iter_queries(args['data_dir'], params['input_size'], 'train')), k=args['n_images']),
        loader.iter_pretraining_renders        (args['data_dir'], params['input_size'], 'segments')))

