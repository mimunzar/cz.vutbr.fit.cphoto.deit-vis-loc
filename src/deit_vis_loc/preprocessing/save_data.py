#!/usr/bin/env python3

import argparse
import functools as ft
import itertools as it
import os
import pickle
import sys

import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.preprocessing.dataset_im as dataset_im
import src.deit_vis_loc.preprocessing.dataset_render as dataset_render


def parse_args(args_it):
    parser = argparse.ArgumentParser(
            description='Creates descriptions of renders and their images')
    parser.add_argument('--geopose-dir', help='The directory of GeoPose3K dataset',
            required=True)
    parser.add_argument('--renders-dir', help='The directory of GeoPose3K Renders dataset',
            required=True)
    return vars(parser.parse_args(args_it))


def im_path_in_renders(dpath, im):
    return os.path.join(dpath, 'query_original_result' , im['name'] + '.jpg')


def render_path_in_renders(dpath, im):
    return os.path.join(dpath, 'database_segments' , im['name'] + '_segments.png')


def add_path(fn_render_path, im):
    im['path'] = fn_render_path(im)
    return im


def write_im_dataset(dpath, fname, im_it):
    with open(os.path.join(dpath, fname), 'wb') as f:
        saver = lambda x: pickle.dump(x, f)
        util.dorun(map(saver, im_it))
        saver('EOF')


def dataset_membership(train_it, val_it, test_it, desc):
    if desc['name'] in train_it: return ('TRAIN', desc)
    if desc['name'] in val_it  : return ('VAL'  , desc)
    if desc['name'] in test_it : return ('TEST' , desc)
    raise ValueError(f'Failed to find {desc["name"]} in datasets')


def iter_render_list(dpath, name):
    fpath = os.path.join(dpath, 'query_original_result', name)
    return (util.first(os.path.splitext(l.strip())) for l in open(fpath))


if '__main__' == __name__:
    args = parse_args(sys.argv[1:])
    renders_dir = os.path.expanduser(args['renders_dir'])
    geopose_dir = os.path.expanduser(args['geopose_dir'])
    membership  = ft.partial(dataset_membership,
        set(iter_render_list(renders_dir, 'train.txt')),
        set(iter_render_list(renders_dir, 'val.txt')),
        set(iter_render_list(renders_dir, 'test.txt')))

    preprocess_dir  = os.path.join(renders_dir, '.preprocessed')
    os.makedirs(preprocess_dir, exist_ok=True)
    dataset_writers = {
        'TRAIN' : ft.partial(write_im_dataset, preprocess_dir, 'train.bin'),
        'VAL'   : ft.partial(write_im_dataset, preprocess_dir, 'val.bin'),
        'TEST'  : ft.partial(write_im_dataset, preprocess_dir, 'test.bin'),
    }

    path_to_im   = ft.partial(add_path, ft.partial(im_path_in_renders, renders_dir))
    parsed_im_it = map(membership, dataset_im.parse(geopose_dir))
    for k, g in it.groupby(sorted(parsed_im_it, key=util.first), key=util.first):
        dataset_writers[k](map(path_to_im, map(util.second, g)))

    add_render_path = ft.partial(add_path, ft.partial(render_path_in_renders, renders_dir))
    write_renders   = ft.partial(write_im_dataset, preprocess_dir, 'renders.bin')
    write_renders(map(add_render_path, dataset_render.parse(renders_dir)))

