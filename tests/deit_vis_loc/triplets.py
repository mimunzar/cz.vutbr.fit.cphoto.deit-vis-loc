#!/usr/bin/env python3

import argparse
import functools as ft
import random
import sys

import matplotlib.pyplot as pyplot
import torch
import torch.hub

import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.crosslocate as crosslocate
import src.deit_vis_loc.training.zoo as zoo
import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.libs.plot as plot
from tests.deit_vis_loc.params import PRETRAINING_PARAMS



def iter_plotted_triplets(triplet_it):
    triplet_it = tuple(triplet_it)
    figure     = pyplot.figure(tight_layout=True)
    plot.im_grid(figure, figure.add_gridspec(nrows=len(triplet_it), ncols=3), triplet_it)
    return triplet_it


def iter_im_triplet(model, params, rd_it, im_it):
    rd_it  = tuple(rd_it)
    iter_d = ft.partial(crosslocate.iter_desc, model)
    with torch.no_grad():
        iter_loc_trp = ft.partial(crosslocate.iter_im_localestriplets,
                params, iter_d, crosslocate.make_mem_iter_desc(iter_d, rd_it))
        yield from map(util.second, iter_loc_trp(rd_it, im_it))


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='The model identifier',
            required=True, choices=[
                'deit_base_patch16_224',
                'deit_base_patch16_384',
                'deit_small_patch16_224',
                'deit_tiny_patch16_224'])
    parser.add_argument('--data-dir', help='The path to dataset',
            required=True, metavar='DIR')
    parser.add_argument('--input-size', help='The resolution of input images',
            required=True, type=int, metavar='INT')
    parser.add_argument('--modality', help='The modality of images',
            required=True, choices=['segments', 'silhouettes', 'depth'])
    parser.add_argument('--scale-by-fov', help='When set scales images by their FOV',
            required=False, action="store_true")
    parser.add_argument('--device', help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--gpu-imcap',   help='The amount of images to fit on GPU',
            required=False, type=int, metavar='INT', default=100)
    parser.add_argument('--n-images', help='The number of images to sample from the dataset',
            required=True, type=int, metavar='INT')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    device = args['device']
    im_it  = random.sample(tuple(loader.iter_queries('train', **args)), k=args['n_images'])
    rd_it  = loader.iter_renders_pretraining(**args)
    model  = {
        'net'       : zoo.new(args['model_name']).to(device),
        'device'    : device,
        'gpu_imcap' : args['gpu_imcap']
    }
    result = map(iter_plotted_triplets,
            iter_im_triplet(model, PRETRAINING_PARAMS, rd_it, im_it))

