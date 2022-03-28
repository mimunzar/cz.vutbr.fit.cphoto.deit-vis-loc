#!/usr/bin/env python3

import argparse
import functools as ft
import itertools as it
import random
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt
import torch.hub

import src.deit_vis_loc.preprocessing.load_data as load_data
import src.deit_vis_loc.training as training
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',    help='The amount of images to sample from the Dataset',
            required=True, type=int, metavar='INT')
    parser.add_argument('--device',      help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


def plot_im_on_axis(im, axis):
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(im['path']))
    return im


def iter_plot_triplets(triplets_it):
    figure   = mpplt.figure(tight_layout=True)
    grid     = figure.add_gridspec(nrows=len(triplets_it), ncols=3)
    plot_im  = lambda r, c, im: plot_im_on_axis(im, figure.add_subplot(grid[r, c]))
    plot_row = lambda r, im_it: tuple(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    return tuple(it.starmap(plot_row, enumerate(triplets_it)))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    im_it  = tuple(load_data.iter_im_data(args['dataset_dir'], 'train.bin'))
    rd_it  = load_data.iter_im_data(args['dataset_dir'], 'renders.bin')
    params = {
        'deit_model' : 'deit_tiny_patch16_224',
        'input_size' : 224,
        'margin'     : 0.2,
        'n_triplets' : 5,
        'positives'  : {
            'dist_m'            : 100,
            'dist_tolerance_m'  : 10,
            'yaw_deg'           : 15,
            'yaw_tolerance_deg' : 1,
        },
        'negatives'  : {
            'dist_m'           : 200,
            'dist_tolerance_m' : 10,
        }
    }

    net = torch.hub.load('facebookresearch/deit:main', params['deit_model'], pretrained=True)
    net.to(args['device'])

    trans = training.make_im_transform(args['device'], params['input_size'])
    fwd   = ft.partial(training.forward, net, trans)
    tp_it = training.iter_n_hard_triplets(params['n_triplets'],
        ft.partial(training.iter_triplets,
            ft.partial(training.iter_pos_renders, params['positives']),
            ft.partial(training.iter_neg_renders, params['negatives'])),
        ft.partial(training.triplet_loss, params['margin'], util.memoize(fwd)),
        random.sample(im_it, k=args['n_images']),  rd_it)
    result = map(iter_plot_triplets, tp_it)

