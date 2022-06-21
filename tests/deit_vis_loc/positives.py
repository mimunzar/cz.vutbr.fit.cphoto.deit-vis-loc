#!/usr/bin/env python3

import argparse
import functools as ft
import itertools as it
import math as ma
import random
import sys

import matplotlib.pyplot as pyplot
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.crosslocate as crosslocate
import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.libs.plot as plot
from tests.deit_vis_loc.params import SPARSE_PARAMS



def iter_im_positives(params, im_it, rd_it):
    def im_positives(rd_it, im):
        return (im, filter(crosslocate.make_is_posrender(params, im), rd_it))
    return map(ft.partial(im_positives, tuple(rd_it)), im_it)


def iter_plot_positives(im, renders_it):
    renders_it = tuple(renders_it)
    fig   = pyplot.figure(tight_layout=True)
    grid  = GridSpec(1, 2, figure=fig)
    ncols = ma.floor(ma.sqrt(len(renders_it)))
    nrows = ma.ceil(len(renders_it)/ncols)
    rgrid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=grid[1])
    plot.im_on_axis(im, fig.add_subplot(grid[0]))
    plot.im_grid(fig, rgrid, util.partition(ncols, renders_it))
    return (im, renders_it)


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='The path to the dataset',
            required=True, metavar='DIR')
    parser.add_argument('--input-size', help='The resolution of input images',
            required=True, type=int, metavar='INT')
    parser.add_argument('--modality', help='The modality of images',
            required=True, choices=['segments', 'silhouettes', 'depth'])
    parser.add_argument('--scale-by-fov', help='When set scales images by their FOV',
            required=False, action="store_true")
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args  = parse_args(sys.argv[1:])
    im_it = tuple(loader.iter_queries('train', **args))
    rd_it = loader.iter_renders_sparse(**args)
    result = it.starmap(iter_plot_positives, iter_im_positives(
        SPARSE_PARAMS['positives'], random.sample(im_it, k=len(im_it)), rd_it))

