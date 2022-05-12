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
    parser.add_argument('--geopose-dir', help='The path to Geopose Dataset',
            required=True, metavar='DIR')
    parser.add_argument('--output-dir',  help='The output directory for Preprocessed Dataset',
            required=True, metavar='DIR')
    parser.add_argument('--resolution',  help='The image output resolution',
            required=True, metavar='INT', type=int)
    return vars(parser.parse_args(args_it))


def save_processed(im_dir, meta_f, data):
    im, meta = data
    im_path  = os.path.join(im_dir, f'{meta["name"]}.jpg')
    im.save(im_path)
    pickle.dump(meta, meta_f)
    return (im_path, meta)


def pad_to_square(res, im):
    n_im = Image.new('RGB', (res, res))
    n_im.paste(im.convert('RGB'), [(res - x)//2 for x in im.size])
    return n_im


def resize_keep_ratio(res, im):
    ratio = res/max(im.size)
    return im.resize([int(ratio*x) for x in im.size], Image.BICUBIC)


def process_im(res, path):
    return pad_to_square(res, resize_keep_ratio(res, Image.open(path)))


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


def process_geo_dir(resolution, dpath):
    im_path = os.path.join(dpath, 'photo.jpg')
    if not os.path.exists(im_path):
        im_path = os.path.join(dpath, 'photo.jpeg')
    return (process_im(resolution, im_path), parse_metafile(os.path.join(dpath, 'info.txt')))


def print_progress(total, data):
    done, *_ = data
    prog_str = log.fmt_bar(bar_width=50, total=total, curr=done)
    print(prog_str.center(log.LINE_WIDTH), end='\n' if total == done else '\r', flush=True)


if '__main__' == __name__:
    args       = parse_args(sys.argv[1:])
    geo_dir    = args['geopose_dir']
    resolution = args['resolution']

    query_dir    = os.path.join(args['output_dir'], 'queries')
    query_im_dir = os.path.join(query_dir, str(resolution))
    geo_name_it  = tuple(util.second(util.first(os.walk(geo_dir))))
    prog_printer = ft.partial(print_progress, len(geo_name_it))

    os.makedirs(query_im_dir, exist_ok=True)
    with open(os.path.join(query_dir, 'meta.bin'), 'wb') as meta_f:
        prog_printer((0, None))
        util.dorun(
            map(prog_printer, enumerate(
                map(ft.partial(save_processed, query_im_dir, meta_f),
                    map(ft.partial(process_geo_dir, resolution),
                        map(ft.partial(os.path.join, geo_dir), geo_name_it))), 1)))
        pickle.dump('EOF', meta_f)

