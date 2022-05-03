#!/usr/bin/env python3

import argparse
import functools as ft
import itertools as it
import random
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt

import src.deit_vis_loc.preprocessing.load_data as load_data
import src.deit_vis_loc.training.locate as locate
import src.deit_vis_loc.training.model as model
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


def iter_plot_mined_triplets(mined):
    figure   = mpplt.figure(tight_layout=True)
    grid     = figure.add_gridspec(nrows=len(mined['triplets']), ncols=3)
    plot_im  = lambda r, c, im: plot_im_on_axis(im, figure.add_subplot(grid[r, c]))
    plot_row = lambda r, im_it: tuple(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    return {**mined, 'triplets': tuple(it.starmap(plot_row, enumerate(mined['triplets'])))}


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    im_it  = load_data.iter_im_data(args['dataset_dir'], 'train.bin')
    rd_it  = load_data.iter_im_data(args['dataset_dir'], 'renders.bin')
    params = {
        'deit_model' : 'deit_tiny_patch16_224',
        'input_size' : 224,
        'margin'     : 0.1,
        'n_triplets' : 5,
        'positives'  : {
            'dist_m'      : 100,
            'dist_tol_m'  : 10,
            'yaw_deg'     : 15,
            'yaw_tol_deg' : 1,
        },
        'negatives'  : {
            'dist_m'     : 200,
            'dist_tol_m' : 10,
        }
    }

    net       = model.load(params['deit_model']).to(args['device']).to(args['device'])
    transform = locate.make_load_im(args['device'], params['input_size'])
    forward   = util.compose(net, transform)
    mined_it  = locate.iter_mined_triplets(params['n_triplets'],
        ft.partial(locate.iter_triplets,
            ft.partial(locate.iter_pos_renders, params['positives']),
            ft.partial(locate.iter_neg_renders, params['negatives'])),
        ft.partial(locate.triplet_loss, params['margin'], util.memoize(forward)),
        random.sample(tuple(im_it), k=args['n_images']),  rd_it)
    result = map(iter_plot_mined_triplets, mined_it)

