#!/usr/bin/env python3

import argparse
import functools as ft
import itertools as it
import math as ma
import random
import sys

import matplotlib.gridspec as mpgrid
import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt

import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.locate as locate
import src.deit_vis_loc.training.model as model
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='The path to dataset',
            required=True, metavar='DIR')
    parser.add_argument('--n-images', help='The amount of images to sample from the Dataset',
            required=True, type=int, metavar='INT')
    parser.add_argument('--resolution', help='The resolution of output images',
            required=False, metavar='INT', type=int, default=224)
    parser.add_argument('--device',   help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


def plot_im_on_axis(im, axis):
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(im['path']))
    return im


def plot_im_grid(figure, grid, im_row_it):
    plot_im  = lambda r, c, im: plot_im_on_axis(im, figure.add_subplot(grid[r, c]))
    plot_row = lambda r, im_it: tuple(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    util.dorun(it.starmap(plot_row, enumerate(im_row_it)))


def iter_plotted_triplets(triplet_it):
    triplet_it = tuple(triplet_it)
    figure     = mpplt.figure(tight_layout=True)
    plot_im_grid(figure, figure.add_gridspec(nrows=len(triplet_it), ncols=3), triplet_it)
    return triplet_it


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
    params        = {'deit_model': 'deit_tiny_patch16_224', 'input_size': res, **PRETRAINING_PARAMS}

    net    = model.load(params['deit_model']).to(device)
    fwd    = util.compose(net, ft.partial(locate.load_im, device))
    result = map(iter_plotted_triplets, locate.iter_im_triplets(device, params, fwd,
        random.sample(tuple(loader.iter_queries(data_dir, res, 'train')), k=args['n_images']),
        loader.iter_pretraining_renders        (data_dir, res, 'segments')))

