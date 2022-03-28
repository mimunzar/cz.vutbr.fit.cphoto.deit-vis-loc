#!/usr/bin/env python3

import argparse
import functools as ft
import itertools as it
import random
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt

import src.deit_vis_loc.preprocessing.load_data as load_data
import src.deit_vis_loc.training as training
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',    help='The amount of images to sample from the Dataset',
            required=True, type=int, metavar='INT')
    return vars(parser.parse_args(args_it))


def iter_triplets(params, im_it, renders_it):
    iter_pos = ft.partial(training.iter_pos_renders, params['positives'])
    iter_neg = ft.partial(training.iter_neg_renders, params['negatives'])
    return training.iter_triplets(iter_pos, iter_neg, im_it, renders_it)


def plot_im_on_axis(im, axis):
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(im['path']))
    return im


def iter_plot_triplets(triplets_it):
    triplets_it = tuple(util.take(5, triplets_it))
    figure   = mpplt.figure(tight_layout=True)
    grid     = figure.add_gridspec(nrows=len(triplets_it), ncols=3)
    plot_im  = lambda r, c, im: plot_im_on_axis(im, figure.add_subplot(grid[r, c]))
    plot_row = lambda r, im_it: tuple(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    return tuple(it.starmap(plot_row, enumerate(triplets_it)))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    im_it  = tuple(load_data.iter_im_data(args['dataset_dir'], 'train.bin'))
    result = map(iter_plot_triplets, iter_triplets({
            'positives': {
                'dist_m'            : 100,
                'dist_tolerance_m'  : 10,
                'yaw_deg'           : 15,
                'yaw_tolerance_deg' : 1,
            },
            'negatives': {
                'dist_m'           : 200,
                'dist_tolerance_m' : 10,
            }
        },
        random.sample(im_it, k=args['n_images']),
        load_data.iter_im_data(args['dataset_dir'], 'renders.bin')))

