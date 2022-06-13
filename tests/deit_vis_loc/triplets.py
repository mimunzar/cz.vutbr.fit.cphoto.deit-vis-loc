#!/usr/bin/env python3

import argparse
import functools as ft
import itertools as it
import random
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt
import torch

import tests.deit_vis_loc.commons as commons
import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.crosslocate as crosslocate
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


def iter_im_triplet(model, params, rd_it, im_it):
    rd_it   = tuple(rd_it)
    iter_d  = ft.partial(crosslocate.iter_desc, params['gpu_imcap'], model)
    with torch.no_grad():
        iter_loc_trp = ft.partial(crosslocate.iter_im_localestriplets,
                params, iter_d, crosslocate.make_mem_iter_desc(iter_d, rd_it))
        yield from map(util.second, iter_loc_trp(rd_it, im_it))


if '__main__' == __name__:
    args       = parse_args(sys.argv[1:])
    data_dir   = args['data_dir']
    device     = args['device']
    resolution = args['resolution']

    params = {
        **commons.PRETRAINING_PARAMS,
        'deit_model': 'deit_tiny_patch16_224',
        'input_size': resolution,
        'gpu_imcap' : 100,
    }
    im_it  = random.sample(tuple(
        loader.iter_queries(data_dir, resolution, 'train')), k=args['n_images'])
    rd_it  = loader.iter_pretraining_renders(data_dir, resolution, 'segments')
    model  = {'net': model.new(params['deit_model']).to(device), 'device': device}
    result = map(iter_plotted_triplets, iter_im_triplet(model, params, rd_it, im_it))

