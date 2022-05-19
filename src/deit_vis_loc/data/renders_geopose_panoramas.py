#!/usr/bin/env python3

import collections as cl
import functools as ft
import itertools as it
import os
import pickle

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


def process_im(resolution, im):
    proc_im = util.compose(
        ft.partial(commons.pad_to_square,     resolution),
        ft.partial(commons.resize_keep_ratio, resolution),
    )
    return proc_im(commons.load_im(im))


def process_render(data_dir, resolution, meta):
    im_path = os.path.join(os.path.join(data_dir, f'{meta["name"]}_segments.png'))
    return (process_im(resolution, im_path), meta)


def print_progress(total, data):
    done, *_ = data
    prog_str = log.fmt_bar(bar_width=50, total=total, curr=done)
    print(prog_str.center(log.LINE_WIDTH), end='\n' if total == done else '\r', flush=True)


def dataset_exists(output_dir, modality, resolution, **_):
    out_dir       = os.path.expanduser(output_dir)
    render_dir    = os.path.join(out_dir, 'renders', 'sparse', modality)
    render_im_dir = os.path.join(render_dir, str(resolution))
    return os.path.exists(render_im_dir)


def write_dataset(sparse_dir, output_dir, modality, resolution, n_images, **_):
    sparse_dir = os.path.expanduser(sparse_dir)
    data_dir   = os.path.join(sparse_dir, 'sparse_database', f'database_{modality}')
    info_fpath = os.path.join(data_dir, 'datasetInfoClean.csv')

    out_dir       = os.path.expanduser(output_dir)
    render_dir    = os.path.join(out_dir, 'renders', 'sparse', modality)
    render_im_dir = os.path.join(render_dir, str(resolution))
    os.makedirs(render_im_dir, exist_ok=True)

    rd_it        = util.take(n_images, commons.iter_csv_file(parse_line, info_fpath))
    prog_printer = ft.partial(print_progress,
        sum(map(util.first, zip(it.repeat(1), util.take(n_images, open(info_fpath))))))

    with open(os.path.join(render_dir, 'meta.bin'), 'wb') as meta_f:
        prog_printer((0, None))
        util.dorun(
            map(prog_printer, enumerate(
                map(ft.partial(save_processed, render_im_dir, meta_f),
                    map(ft.partial(process_render, data_dir, resolution), rd_it)), 1)))
        pickle.dump('EOF', meta_f)

