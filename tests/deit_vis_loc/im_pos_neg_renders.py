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

import src.deit_vis_loc.data.load_data as load_data
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


def plot_im_grid(fig, grid, im_it):
    im_it    = tuple(im_it)
    ncols    = ma.floor(ma.sqrt(len(im_it)))
    nrows    = ma.ceil(len(im_it)/ncols)
    im_grid  = mpgrid.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=grid)
    plot_im  = lambda r, c, im: plot_im_on_axis(im, fig.add_subplot(im_grid[r, c]))
    plot_row = lambda r, im_it: util.dorun(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    util.dorun(it.starmap(plot_row, enumerate(util.partition(ncols, im_it, strict=False))))


def iter_plot_im_pos_neg_renders(im_it, pos_it, neg_it):
    pos_it = tuple(pos_it)
    neg_it = tuple(neg_it)
    fig    = mpplt.figure(tight_layout=True)
    grid   = mpgrid.GridSpec(nrows=1, ncols=2, figure=fig)
    lgrid  = mpgrid.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=grid[0])
    plot_im_on_axis(util.first(im_it),  fig.add_subplot(lgrid[0]))
    plot_im_on_axis(util.first(pos_it), fig.add_subplot(lgrid[1]))
    plot_im_grid(fig, grid[1], neg_it)
    return (im_it, pos_it, neg_it)


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    im_it  = tuple(load_data.iter_im_data(args['dataset_dir'], 'train.bin'))
    rd_it  = tuple(load_data.iter_im_data(args['dataset_dir'], 'renders.bin'))
    params = {
        'deit_model'    : 'deit_tiny_patch16_224',
        'input_size'    : 224,
        'partition_size': 200,
        'positives': {
            'samples'     : 2,
            'dist_m'      : 100,
            'dist_tol_m'  : 10,
            'yaw_deg'     : 15,
            'yaw_tol_deg' : 1,
        },
        'negatives': {
            'samples'     : 5,
            'dist_m'      : 100,
            'dist_tol_m'  : 10,
            'yaw_deg'     : 15,
            'yaw_tol_deg' : 1,
        }
    }
    net    = model.load(params['deit_model']).to(args['device']).to(args['device'])
    fwd    = util.compose(net, locate.make_load_im(args['device'], params['input_size']))
    result = it.starmap(iter_plot_im_pos_neg_renders,
        locate.iter_im_pos_neg_renders(args['device'], params, fwd, im_it, rd_it))

