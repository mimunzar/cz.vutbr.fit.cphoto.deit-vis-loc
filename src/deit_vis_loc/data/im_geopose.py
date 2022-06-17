#!/usr/bin/env python3

import collections as cl
import functools as ft
import os
import pickle
from PIL import Image, ImageOps

import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.data.commons as commons


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


def process_geo_dir(fn_membership, fn_transform_im, dpath):
    metadata = parse_metafile(os.path.join(dpath, 'info.txt'))
    im_path  = os.path.join(dpath, 'photo.jpeg')
    if not os.path.exists(im_path):
        im_path = os.path.join(dpath, 'photo.jpg')
    return (fn_transform_im(im_path), (fn_membership(metadata), metadata))


def print_progress(total, data):
    done = util.first(data)
    prog = f'{log.fmt_bar(50, total, done)} {log.fmt_fraction(total, done)}'
    print(prog.center(log.LINE_WIDTH), end='\n' if total == done else '\r',
            flush=True)


def make_im_transform(input_size):
    im_name    = util.compose(util.second, os.path.split, os.path.dirname)
    wrong_exif = {'flickr_sge_3116199276_e23b30b95e_3268_31934892@N06'}
    #^ These images when rotated according to theyr Exif tag ends up being  with
    #  a  wrong  orientation.   Don't  apply  the  rotation  to  these   images.
    def open_adjust_orientation(fpath):
        im = Image.open(fpath)
        return im if im_name(fpath) in wrong_exif else ImageOps.exif_transpose(im)
    return util.compose(
        ft.partial(commons.pad_to_square,     input_size),
        ft.partial(commons.resize_keep_ratio, input_size),
        open_adjust_orientation)


def iter_query_list(dpath, name):
    strip_ext = util.compose(util.first, os.path.splitext)
    return (strip_ext(l.strip()) for l in open(os.path.join(dpath, name)))


def dataset_membership(train_it, val_it, test_it, desc):
    name = desc['name']
    if name in train_it: return 'TRAIN'
    if name in val_it  : return 'VAL'
    if name in test_it : return 'TEST'
    raise ValueError(f'Failed to find {name} in datasets')


def dataset_exists(output_dir, input_size, **_):
    output_dir   = os.path.expanduser(output_dir)
    query_dir    = os.path.join(output_dir, 'queries')
    query_im_dir = os.path.join(query_dir, str(input_size))
    return os.path.exists(query_im_dir)


def write_dataset(geopose_dir, sparse_dir, output_dir, input_size, n_images, **_):
    output_dir   = os.path.expanduser(output_dir)
    query_dir    = os.path.join(output_dir, 'queries')
    query_im_dir = os.path.join(query_dir, str(input_size))
    os.makedirs(query_im_dir, exist_ok=True)

    sparse_dir   = os.path.expanduser(sparse_dir)
    sparse_q_dir = os.path.join(sparse_dir, 'sparse_queries', 'query_original_result')
    membership   = ft.partial(dataset_membership,
        set(iter_query_list(sparse_q_dir, 'train.txt')),
        set(iter_query_list(sparse_q_dir, 'val.txt')),
        set(iter_query_list(sparse_q_dir, 'test.txt')))
    im_transform = make_im_transform(input_size)

    train_f      = open(os.path.join(query_dir, 'train.bin'), 'wb')
    val_f        = open(os.path.join(query_dir, 'val.bin'),   'wb')
    test_f       = open(os.path.join(query_dir, 'test.bin'),  'wb')
    meta_writers = {
        'TRAIN': lambda m: pickle.dump(m, train_f),
        'VAL'  : lambda m: pickle.dump(m, val_f),
        'TEST' : lambda m: pickle.dump(m, test_f),
    }

    geopose_dir  = os.path.expanduser(geopose_dir)
    geo_name_it  = tuple(util.take(n_images,
        util.second(util.first(os.walk(geopose_dir)))))
    prog_printer = ft.partial(print_progress, len(geo_name_it))

    prog_printer((0, None))
    util.dorun(
        map(prog_printer, enumerate(
            map(ft.partial(save_processed, query_im_dir, meta_writers),
                map(ft.partial(process_geo_dir, membership, im_transform),
                    map(ft.partial(os.path.join, geopose_dir), geo_name_it))), 1)))

    pickle.dump('EOF', train_f)
    pickle.dump('EOF', val_f)
    pickle.dump('EOF', test_f)

    train_f.close()
    val_f  .close()
    test_f .close()

