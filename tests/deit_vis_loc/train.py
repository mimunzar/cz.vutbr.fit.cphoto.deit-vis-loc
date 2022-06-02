#!/usr/bin/env python3

import argparse
import sys

import torch.optim

import src.deit_vis_loc.libs.util as util
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
    parser.add_argument('--resolution',  help='The resolution of output images',
            required=True, metavar='INT', type=int)
    parser.add_argument('--device',      help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args       = parse_args(sys.argv[1:])
    device     = args['device']
    data_dir   = args['data_dir']
    resolution = args['resolution']
    n_images   = args['n_images']

    params = {
        **commons.PRETRAINING_PARAMS,
        'deit_model'       : 'deit_tiny_patch16_224',
        'input_size'       : resolution,
        'margin'           : 0.1,
        'lr'               : 1e-3,
        'batch_size'       : 5,
        'mine_every_epoch' : 2,
    }
    net   = model.load(params['deit_model']).to(device)
    model = {
        'device': device,
        'net'   : net,
        'optim' : torch.optim.SGD(net.parameters(), params['lr'], momentum=0.9)
    }
    tim_it = util.take(n_images, loader.iter_queries(data_dir, params['input_size'], 'train'))
    vim_it = util.take(n_images, loader.iter_queries(data_dir, params['input_size'], 'val'))
    rd_it  = loader.iter_pretraining_renders(data_dir, params['input_size'], 'segments')
    result = locate.iter_trainingepoch(model, params, vim_it, tim_it, rd_it)

