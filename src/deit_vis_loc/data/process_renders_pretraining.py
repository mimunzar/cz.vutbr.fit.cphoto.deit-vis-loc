#!/usr/bin/env python3

import argparse
import collections as cl
import functools as ft
import itertools as it
import os
import pickle
import sys
from PIL import Image, ImageOps

import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.data.commons as commons


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse-dir',  help='The path to Sparse dataset',
            required=True, metavar='DIR')
    parser.add_argument('--output-dir',  help='The output directory for resulting dataset',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',    help='The number of images of resulting dataset',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--modality',    help='The modality of images of resulting dataset',
            required=True, choices=['segments', 'silhouettes', 'depth'])
    parser.add_argument('--resolution',  help='The resolution of output images',
            required=True, metavar='INT', type=int)
    return vars(parser.parse_args(args_it))


def save_processed(im_dir, meta_f, data):
    im, meta = data
    im_path  = os.path.join(im_dir, f'{meta["name"]}.jpg')
    im.save(im_path)
    pickle.dump(meta, meta_f)
    return (im_path, meta)


INFO_FILE_FIELDS = cl.OrderedDict({
    'id'        : lambda x: x,
    'query'     : lambda x: x,
    'latitude'  : lambda x: float(x),
    'longitude' : lambda x: float(x),
    'elevation' : lambda x: float(x),
    'yaw'       : lambda x: float(x),
    'pitch'     : lambda x: float(x),
    'roll'      : lambda x: float(x),
    'fov'       : lambda x: float(x),
})

def parse_line(csv_it):
    parsed = commons.parse_into(INFO_FILE_FIELDS, csv_it)
    return {
        'name'      : parsed['query'],
        'query'     : parsed['query'],
        'latitude'  : parsed['latitude'],
        'longitude' : parsed['longitude'],
        'elevation' : parsed['elevation'],
        'yaw'       : parsed['yaw'],
        'pitch'     : parsed['pitch'],
        'roll'      : parsed['roll'],
    }


def process_im(resolution, im):
    proc_im = util.compose(
        ft.partial(commons.pad_to_square,     resolution),
        ft.partial(commons.resize_keep_ratio, resolution),
    )
    return proc_im(commons.load_im(im))


def process_render(data_dir, resolution, meta):
    im_path = os.path.join(os.path.join(data_dir, f'{meta["name"]}.png'))
    return (process_im(resolution, im_path), meta)


def print_progress(total, data):
    done, *_ = data
    prog_str = log.fmt_bar(bar_width=50, total=total, curr=done)
    print(prog_str.center(log.LINE_WIDTH), end='\n' if total == done else '\r', flush=True)


if '__main__' == __name__:
    args       = parse_args(sys.argv[1:])
    resolution = args['resolution']
    modality   = args['modality']

    geo_dir    = os.path.expanduser(args['sparse_dir'])
    data_dir   = os.path.join(geo_dir, 'sparse_queries', f'query_{modality}')
    info_fpath = os.path.join(data_dir, 'datasetInfoClean.csv')

    out_dir       = os.path.expanduser(args['output_dir'])
    render_dir    = os.path.join(out_dir, 'renders', 'pretraining', modality)
    render_im_dir = os.path.join(render_dir, str(resolution))

    rd_it        = util.take(args['n_images'], commons.iter_csv_file(parse_line, info_fpath))
    prog_printer = ft.partial(print_progress,
        sum(map(util.first, zip(it.repeat(1), util.take(args['n_images'], open(info_fpath))))))

    os.makedirs(render_im_dir, exist_ok=True)
    with open(os.path.join(render_dir, 'meta.bin'), 'wb') as meta_f:
        prog_printer((0, None))
        util.dorun(
            map(prog_printer, enumerate(
                map(ft.partial(save_processed, render_im_dir, meta_f),
                    map(ft.partial(process_render, data_dir, resolution), rd_it)), 1)))
        pickle.dump('EOF', meta_f)

