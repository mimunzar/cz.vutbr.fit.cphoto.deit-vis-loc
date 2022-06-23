#!/usr/bin/env python3

import collections as cl
import functools as ft
import os
import pickle
from PIL import Image, ImageOps

import src.deit_vis_loc.libs.image as image
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.data.csv as csv



def save_im_metadata(fn_membership, meta_writers, meta):
    meta_writers[fn_membership(meta)](meta)
    return meta


CAMERA_FIELDS = cl.OrderedDict({
    'yaw'  : lambda x: float(x),
    'pitch': lambda x: float(x),
    'roll' : lambda x: float(x),
})
INFO_FILE_FIELDS = cl.OrderedDict({
    'annotation' : lambda x: x,
    'camera'     : lambda x: csv.values_into(CAMERA_FIELDS, x.split()),
    'latitude'   : lambda x: float(x),
    'longitude'  : lambda x: float(x),
    'elevation'  : lambda x: float(x),
    'FOV'        : lambda x: float(x),
})

def meta_struct(name, content_it):
    parsed = csv.values_into(INFO_FILE_FIELDS, content_it)
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


def parse_metafile(geo_im_dir):
    im_name   = os.path.basename(geo_im_dir)
    info_path = os.path.join(geo_im_dir, 'info.txt')
    try:
        return meta_struct(im_name, (l.strip() for l in open(info_path)))
    except Exception as ex:
        raise ValueError(f'Failed to parse {info_path} ({ex})')


def save_transformed_im(fn_transform_im, out_dir, geo_im_dir):
    meta    = parse_metafile(geo_im_dir)
    im_path = os.path.join(geo_im_dir, 'photo.jpeg')
    if not os.path.exists(im_path):
        im_path = os.path.join(geo_im_dir, 'photo.jpg')
    with Image.open(im_path) as im:
        fn_transform_im(im, meta).save(os.path.join(out_dir, f'{meta["name"]}.jpg'))
    return meta


def print_progress(total, data):
    done = util.first(data)
    prog = f'{log.fmt_bar(50, total, done)} {log.fmt_fraction(total, done)}'
    print(prog.center(log.LINE_WIDTH), end='\n' if total == done else '\r', flush=True)


def make_im_transform(input_size, scale_by_fov):
    wrong_exif_it = {'flickr_sge_3116199276_e23b30b95e_3268_31934892@N06'}
    #^ These images when rotated according to their Exif tag ends up being  with
    # a wrong orientation.  Don't apply  rotation  operation  to  these  images.
    def adjust_orientation(im, im_name):
        return im if im_name in wrong_exif_it else ImageOps.exif_transpose(im)
    def im_transform_fov(im, meta):
        return util.compose(
            ft.partial(image.pad_to_square, input_size),
            ft.partial(image.center_crop,   input_size),
            ft.partial(image.scale_by_fov,  input_size, meta['FOV']),
            adjust_orientation)(im, meta['name'])
    def im_transform(im, meta):
        return util.compose(
            ft.partial(image.pad_to_square, input_size),
            ft.partial(image.scale_to_fit,  input_size),
            adjust_orientation)(im, meta['name'])
    return im_transform_fov if scale_by_fov else im_transform


def iter_query_list(dpath, name):
    strip_ext = util.compose(util.first, os.path.splitext)
    return (strip_ext(l.strip()) for l in open(os.path.join(dpath, name)))


def dataset_membership(train_it, val_it, test_it, meta):
    name = meta['name']
    if name in train_it: return 'TRAIN'
    if name in val_it  : return 'VAL'
    if name in test_it : return 'TEST'
    raise ValueError(f'Failed to find {name} in datasets')


def data_suffix(input_size, scale_by_fov):
    return os.path.join('data_fov' if scale_by_fov else 'data', str(input_size))


def dataset_exists(output_dir, input_size, scale_by_fov, **_):
    output_dir = os.path.expanduser(output_dir)
    data_dir   = os.path.join(output_dir, 'queries', data_suffix(input_size, scale_by_fov))
    return (os.path.exists(data_dir), data_dir)


def write_dataset(geopose_dir,
        sparse_dir, output_dir, input_size, n_images, scale_by_fov, **_):
    meta_dir   = os.path.join(os.path.expanduser(output_dir), 'queries')
    im_dir     = os.path.join(meta_dir, data_suffix(input_size, scale_by_fov))
    os.makedirs(im_dir, exist_ok=True)

    sparse_dir   = os.path.expanduser(sparse_dir)
    sparse_q_dir = os.path.join(sparse_dir, 'sparse_queries', 'query_original_result')
    membership   = ft.partial(dataset_membership,
        set(iter_query_list(sparse_q_dir, 'train.txt')),
        set(iter_query_list(sparse_q_dir, 'val.txt')),
        set(iter_query_list(sparse_q_dir, 'test.txt')))
    im_transform = make_im_transform(input_size, scale_by_fov)

    train_f      = open(os.path.join(meta_dir, 'train.bin'), 'wb')
    val_f        = open(os.path.join(meta_dir, 'val.bin'),   'wb')
    test_f       = open(os.path.join(meta_dir, 'test.bin'),  'wb')
    meta_writers = {
        'TRAIN': lambda m: pickle.dump(m, train_f),
        'VAL'  : lambda m: pickle.dump(m, val_f),
        'TEST' : lambda m: pickle.dump(m, test_f),
    }

    geopose_dir  = os.path.expanduser(geopose_dir)
    geo_names_it = tuple(util.take(n_images,
        util.second(util.first(os.walk(geopose_dir)))))
    prog_printer = ft.partial(print_progress, len(geo_names_it))

    prog_printer((0, None))
    util.dorun(
        map(prog_printer, enumerate(
            map(ft.partial(save_im_metadata, membership, meta_writers),
                map(ft.partial(save_transformed_im, im_transform, im_dir),
                    map(ft.partial(os.path.join, geopose_dir), geo_names_it))), 1)))

    pickle.dump('EOF', train_f)
    pickle.dump('EOF', val_f)
    pickle.dump('EOF', test_f)

    train_f.close()
    val_f  .close()
    test_f .close()

