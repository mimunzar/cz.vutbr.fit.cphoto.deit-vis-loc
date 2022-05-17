#!/usr/bin/env python3

import argparse
import functools as ft
import sys

import torch.optim

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


PRETRAINING_PARAMS =  {
    'positives': {
        'samples'     : 1,
        'dist_m'      : 0,
        'dist_tol_m'  : 1,
        'yaw_deg'     : 0,
        'yaw_tol_deg' : 1,
   },
   'negatives': {
       'samples'     : 5,
       'dist_m'      : 2000,
       'dist_tol_m'  : 10,
   }
}

SPARSE_PARAMS = {
    'positives': {
        'samples'     : 1,
        'dist_m'      : 20,
        'dist_tol_m'  : 1,
        'yaw_deg'     : 15,
        'yaw_tol_deg' : 1,
    },
    'negatives': {
        'samples'     : 5,
        'dist_m'      : 2000,
        'dist_tol_m'  : 10,
    }
}



if '__main__' == __name__:
    args          = parse_args(sys.argv[1:])
    data_dir, res = util.pluck(['data_dir', 'resolution'], args)
    n_im, device  = util.pluck(['n_images', 'device'],     args)
    params        = {
        'deit_model' : 'deit_tiny_patch16_224',
        'input_size' : res,
        'margin'     : 0.1,
        'lr'         : 1e-3,
        'mine_every' : 1,
        **PRETRAINING_PARAMS,
    }

    net    = model.load(params['deit_model']).to(device)
    optim  = torch.optim.AdamW(net.parameters(), params['lr'])
    fwd  = util.compose(net, ft.partial(locate.load_im, device))

    result = locate.iter_epoch_im_triplets(device, params, fwd,
            loader.iter_queries            (data_dir, res, 'train'),
            loader.iter_pretraining_renders(data_dir, res, 'segments'))

    # result = locate.iter_epoch_im_loss({
    #         'device' : device,
    #         'net'    : net,
    #         'optim'  : optim,
    #     },
    #     params,
    #     loader.iter_queries            (data_dir, res, 'train'),
    #     loader.iter_pretraining_renders(data_dir, res, 'segments'))



