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


def iter_positives(fn_fwd, params, im_it, rd_it):
    rd_it = tuple(rd_it)
    def im_positives(im):
        pos_it  = locate.iter_pos_renders(params, im, rd_it)
        hard_it = locate.iter_hard_pos_renders(fn_fwd, params, im, rd_it)
        return (im, util.first(hard_it), pos_it)
    return map(im_positives, im_it)


def plot_im_on_axis(im, axis):
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(im['path']))
    return im


def iter_plot_positives(im, hard_im, pos_it):
    pos_it = tuple(pos_it)
    fig    = mpplt.figure(tight_layout=True)
    grid   = mpgrid.GridSpec(nrows=1, ncols=2, figure=fig)
    lgrid  = mpgrid.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=grid[0])
    plot_im_on_axis(im,      fig.add_subplot(lgrid[0, 0]))
    plot_im_on_axis(hard_im, fig.add_subplot(lgrid[1, 0]))

    ncols    = ma.floor(ma.sqrt(len(pos_it)))
    nrows    = ma.ceil(len(pos_it)/ncols)
    rgrid    = mpgrid.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=grid[1])
    plot_im  = lambda r, c, im: plot_im_on_axis(im, fig.add_subplot(rgrid[r, c]))
    plot_row = lambda r, im_it: tuple(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    return (im, tuple(util.flatten(it.starmap(plot_row, enumerate(util.partition(ncols, pos_it))))))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    im_it  = random.sample(tuple(load_data.iter_im_data(args['dataset_dir'], 'train.bin')), k=args['n_images'])
    rd_it  = load_data.iter_im_data(args['dataset_dir'], 'renders.bin')
    params = {
        'deit_model' : 'deit_tiny_patch16_224',
        'input_size' : 224,
        'positives': {
            'n_positives' : 1,
            'dist_m'      : 100,
            'dist_tol_m'  : 10,
            'yaw_deg'     : 15,
            'yaw_tol_deg' : 1,
        }
    }
    net    = model.load(params['deit_model']).to(args['device']).to(args['device'])
    fwd    = util.compose(net, locate.make_load_im(args['device'], params['input_size']))
    result = it.starmap(iter_plot_positives, iter_positives(fwd, params['positives'], im_it, rd_it))

