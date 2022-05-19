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

import tests.deit_vis_loc.commons as commons
import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.locate as locate
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   help='The path to the dataset',
            required=True, metavar='DIR')
    parser.add_argument('--resolution', help='The resolution of output images',
            required=False, metavar='INT', type=int, default=224)
    return vars(parser.parse_args(args_it))


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


def iter_positives(params, im_it, renders_it):
    renders_it = tuple(renders_it)
    return map(lambda im: (im, locate.iter_posrenders(params, im, renders_it)), im_it)


if '__main__' == __name__:
    args  = parse_args(sys.argv[1:])
    im_it = tuple(loader.iter_queries (args['data_dir'], args['resolution'], 'train'))
    rd_it = loader.iter_sparse_renders(args['data_dir'], args['resolution'], 'segments')
    result = it.starmap(iter_plot_positives,
        iter_positives(commons.SPARSE_PARAMS['positives'], random.sample(im_it, k=len(im_it)), rd_it))

