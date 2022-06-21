#!/usr/bin/env python3

import collections as cl
import functools as ft
import os
import pickle
from PIL import Image

import src.deit_vis_loc.data.commons as commons
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util


def save_processed(im_dir, meta_f, data):
    im, meta = data
    im_path  = os.path.join(im_dir, f'{meta["name"]}.jpg')
    im.save(im_path)
    pickle.dump(meta, meta_f)
    return (im_path, meta)


INFO_FILE_FIELDS = cl.OrderedDict({
    'segment'   : lambda x: x,
    'query'     : lambda x: x,
    'latitude'  : lambda x: float(x),
    'longitude' : lambda x: float(x),
    'elevation' : lambda x: float(x),
    'rotation'  : lambda x: float(x),
    'yaw'       : lambda x: float(x),
    'pitch'     : lambda x: float(x),
    'roll'      : lambda x: float(x),
})

def parse_line(csv_it):
    parsed = commons.parse_into(INFO_FILE_FIELDS, csv_it)
    return {
        'name'      : parsed['segment'],
        'query'     : parsed['query'],
        'latitude'  : parsed['latitude'],
        'longitude' : parsed['longitude'],
        'elevation' : parsed['elevation'],
        'yaw'       : parsed['rotation'],
        'pitch'     : 0,
        'roll'      : 0,
    }
    #^ In datasetInfoClean.csv the rotation fields coresponds to yaw angles
    # for geopose images and the pitch and roll are 0.


def process_render(fn_transform_im, data_dir, meta):
    im_path = os.path.join(data_dir, f'{meta["name"]}_segments.png')
    return (fn_transform_im(im_path), meta)


def print_progress(total, data):
    done = util.first(data)
    prog = f'{log.fmt_bar(50, total, done)} {log.fmt_fraction(total, done)}'
    print(prog.center(log.LINE_WIDTH), end='\n' if total == done else '\r',
            flush=True)


def make_im_transform(input_size):
    return util.compose(
        ft.partial(commons.pad_to_square,     input_size),
        ft.partial(commons.resize_keep_ratio, input_size),
        Image.open)


def dataset_exists(output_dir, modality, input_size, **_):
    out_dir       = os.path.expanduser(output_dir)
    render_dir    = os.path.join(out_dir, 'renders', 'sparse', modality)
    render_im_dir = os.path.join(render_dir, str(input_size))
    return os.path.exists(render_im_dir)


def write_dataset(sparse_dir, output_dir, modality, input_size, n_images, **_):
    sparse_dir = os.path.expanduser(sparse_dir)
    data_dir   = os.path.join(sparse_dir, 'sparse_database', f'database_{modality}')
    info_fpath = os.path.join(data_dir, 'datasetInfoClean.csv')

    out_dir       = os.path.expanduser(output_dir)
    render_dir    = os.path.join(out_dir, 'renders', 'sparse', modality)
    render_im_dir = os.path.join(render_dir, str(input_size))
    os.makedirs(render_im_dir, exist_ok=True)

    rd_it        = tuple(util.take(n_images, commons.iter_csv_file(parse_line, info_fpath)))
    prog_printer = ft.partial(print_progress, len(rd_it))
    im_transform = make_im_transform(input_size)
    with open(os.path.join(render_dir, 'meta.bin'), 'wb') as meta_f:
        prog_printer((0, None))
        util.dorun(
            map(prog_printer, enumerate(
                map(ft.partial(save_processed, render_im_dir, meta_f),
                    map(ft.partial(process_render, im_transform, data_dir), rd_it)), 1)))
        pickle.dump('EOF', meta_f)

