#!/usr/bin/env python3

import argparse
import collections as cl
import functools as ft
import os
import pickle
import sys
from PIL import Image

import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.data.commons as commons


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--geopose-dir', help='The path to Geopose dataset',
            required=True, metavar='DIR')
    parser.add_argument('--sparse-dir',  help='The path to Sparse dataset',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',    help='The number of images of resulting dataset',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--output-dir',  help='The output directory for resulting dataset',
            required=True, metavar='DIR')
    parser.add_argument('--resolution',  help='The resolution of output images',
            required=True, metavar='INT', type=int)
    return vars(parser.parse_args(args_it))


def save_processed(im_dir, meta_writers, data):
    im, (member, meta) = data
    im_path            = os.path.join(im_dir, f'{meta["name"]}.jpg')
    meta_writers[member](meta)
    im.save(im_path)
    return (im_path, meta)


CAMERA_FIELDS = cl.OrderedDict({
    'yaw'  : lambda x: float(x),
    'pitch': lambda x: float(x),
    'roll' : lambda x: float(x),
})
INFO_FILE_FIELDS = cl.OrderedDict({
    'annotation' : lambda x: x,
    'camera'     : lambda x: commons.parse_into(CAMERA_FIELDS, x.split()),
    'latitude'   : lambda x: float(x),
    'longitude'  : lambda x: float(x),
    'elevation'  : lambda x: float(x),
    'FOV'        : lambda x: float(x),
})

def parse_meta(name, content_it):
    parsed = commons.parse_into(INFO_FILE_FIELDS, content_it)
    return {
        'name'      : name,
        'latitude'  : parsed['latitude'],
        'longitude' : parsed['longitude'],
        'elevation' : parsed['elevation'],
        'yaw'       : parsed['camera']['yaw'],
        'pitch'     : parsed['camera']['pitch'],
        'roll'      : parsed['camera']['roll'],
        'FOV'       : parsed['FOV'],
    }


def parse_metafile(path):
    _, name = os.path.split(util.first(os.path.split(path)))
    try:
        return parse_meta(name, (l.strip() for l in open(path)))
    except Exception as ex:
        raise ValueError(f'Failed to parse {path} ({ex})')


def process_im(resolution, im):
    proc_im = util.compose(
        ft.partial(commons.pad_to_square,     resolution),
        ft.partial(commons.resize_keep_ratio, resolution),
    )
    return proc_im(Image.open(im))


def process_geo_dir(fn_member, resolution, dpath):
    im_path = os.path.join(dpath, 'photo.jpg')
    if not os.path.exists(im_path):
        im_path = os.path.join(dpath, 'photo.jpeg')
    return (process_im(resolution, im_path),
            fn_member(parse_metafile(os.path.join(dpath, 'info.txt'))))


def print_progress(total, data):
    done, *_ = data
    prog_str = log.fmt_bar(bar_width=50, total=total, curr=done)
    print(prog_str.center(log.LINE_WIDTH),
            end='\n' if total == done else '\r', flush=True)


def iter_query_list(dpath, name):
    return (util.first(os.path.splitext(l.strip()))
            for l in open(os.path.join(dpath, name)))


def dataset_membership(train_it, val_it, test_it, desc):
    name = desc['name']
    if name in train_it: return ('TRAIN', desc)
    if name in val_it  : return ('VAL'  , desc)
    if name in test_it : return ('TEST' , desc)
    raise ValueError(f'Failed to find {name} in datasets')


if '__main__' == __name__:
    args       = parse_args(sys.argv[1:])
    resolution = args['resolution']

    out_dir      = os.path.expanduser(args['output_dir'])
    query_dir    = os.path.join(out_dir, 'queries')
    query_im_dir = os.path.join(query_dir, str(resolution))
    os.makedirs(query_im_dir, exist_ok=True)

    sparse_dir   = os.path.expanduser(args['sparse_dir'])
    sparse_q_dir = os.path.join(sparse_dir, 'sparse_queries', 'query_original_result')
    membership   = ft.partial(dataset_membership,
        set(iter_query_list(sparse_q_dir, 'train.txt')),
        set(iter_query_list(sparse_q_dir, 'val.txt')),
        set(iter_query_list(sparse_q_dir, 'test.txt')))

    train_f      = open(os.path.join(query_dir, 'train.bin'), 'wb')
    val_f        = open(os.path.join(query_dir, 'val.bin'),   'wb')
    test_f       = open(os.path.join(query_dir, 'test.bin'),  'wb')
    meta_writers = {
        'TRAIN': lambda m: pickle.dump(m, train_f),
        'VAL'  : lambda m: pickle.dump(m, val_f),
        'TEST' : lambda m: pickle.dump(m, test_f),
    }

    geo_dir      = os.path.expanduser(args['geopose_dir'])
    geo_name_it  = tuple(util.take(args['n_images'], util.second(util.first(os.walk(geo_dir)))))
    prog_printer = ft.partial(print_progress, len(geo_name_it))

    prog_printer((0, None))
    util.dorun(
        map(prog_printer, enumerate(
            map(ft.partial(save_processed, query_im_dir, meta_writers),
                map(ft.partial(process_geo_dir, membership, resolution),
                    map(ft.partial(os.path.join, geo_dir), geo_name_it))), 1)))

    pickle.dump('EOF', train_f)
    pickle.dump('EOF', val_f)
    pickle.dump('EOF', test_f)

    train_f.close()
    val_f  .close()
    test_f .close()

