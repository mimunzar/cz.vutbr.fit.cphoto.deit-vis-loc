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

import src.deit_vis_loc.preprocessing.load_data as load_data
import src.deit_vis_loc.training as training
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    return vars(parser.parse_args(args_it))


def iter_positives(train_params, im_it, renders_it):
    renders_it = tuple(renders_it)
    positives  = lambda im: training.iter_pos_renders(train_params, im, renders_it)
    return map(lambda im: (im, positives(im)), im_it)


def plot_im_on_axis(im, axis):
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(im['path']))
    return im


def iter_plot_positives(im, renders_it):
    renders_it = tuple(renders_it)
    fig   = mpplt.figure(tight_layout=True)
    grid  = mpgrid.GridSpec(1, 2, figure=fig)
    ncols = ma.floor(ma.sqrt(len(renders_it)))
    nrows = ma.ceil(len(renders_it)/ncols)
    rgrid = mpgrid.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=grid[1])

    plot_im_on_axis(im, fig.add_subplot(grid[0]))
    plot_im  = lambda r, c, im: plot_im_on_axis(im, fig.add_subplot(rgrid[r, c]))
    plot_row = lambda r, im_it: tuple(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    return (im, tuple(util.flatten(it.starmap(plot_row, enumerate(util.partition(ncols, renders_it))))))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    pos_it = tuple(iter_positives({
            'dist_m'            : 100,
            'dist_tolerance_m'  : 0,
            'yaw_deg'           : 15,
            'yaw_tolerance_deg' : 1,
        },
        load_data.iter_im_data(args['dataset_dir'], 'train.bin'),
        load_data.iter_im_data(args['dataset_dir'], 'renders.bin')))
    result = it.starmap(iter_plot_positives, random.sample(pos_it, k=len(pos_it)))

