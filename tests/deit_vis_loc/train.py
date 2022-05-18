#!/usr/bin/env python3

import argparse
import random
import sys

import torch.optim

import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.locate as locate
import src.deit_vis_loc.training.model as model
import tests.deit_vis_loc.commons as commons


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',    help='The path to dataset',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',    help='The amount of images to sample',
            required=True, type=int, metavar='INT')
    parser.add_argument('--n-epochs',    help='The number of epochs to iterate',
            required=True, type=int, metavar='INT')
    parser.add_argument('--device',      help='The device to use',
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
        'max_epochs'       : args['n_epochs'],
        'mine_every_epoch' : 2,
        **commons.PRETRAINING_PARAMS,
    }
    net    = model.load(params['deit_model']).to(args['device'])
    result = locate.train({
            'device' : args['device'],
            'net'    : net,
            'optim'  : torch.optim.SGD(net.parameters(), params['lr'], momentum=0.9),
        },
        params,
        random.sample(tuple(loader.iter_queries(args['data_dir'], params['input_size'], 'train')), k=args['n_images']),
        loader.iter_pretraining_renders        (args['data_dir'], params['input_size'], 'segments'))
