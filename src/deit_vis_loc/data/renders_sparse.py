#!/usr/bin/env python3

import collections as cl
import functools as ft
import os
import pickle

import src.deit_vis_loc.data.csv as csv
import src.deit_vis_loc.libs.image as image
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util


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
    parsed = csv.values_into(INFO_FILE_FIELDS, csv_it)
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


def save_im_metadata(meta_f, meta):
    pickle.dump(meta, meta_f)
    return meta


def save_transformed_im(fn_transform_im, modality, in_dir, out_dir, meta):
    im_name  = meta["name"]
    im_ext   = 'exr' if 'depth' == modality else 'png'
    in_path  = os.path.join(in_dir,  f'{im_name}_{modality}.{im_ext}')
    out_path = os.path.join(out_dir, f'{im_name}.{im_ext}')
    image.write(out_path, fn_transform_im(image.read(in_path)))
    return meta


def print_progress(total, data):
    done = util.first(data)
    prog = f'{log.fmt_bar(50, total, done)} {log.fmt_fraction(total, done)}'
    print(prog.center(log.LINE_WIDTH), end='\n' if total == done else '\r', flush=True)


def make_im_transform(input_size):
    return util.compose(
        ft.partial(image.pad_to_square, input_size),
        ft.partial(image.scale_to_fit,  input_size))


def dataset_exists(output_dir, modality, input_size, **_):
    meta_dir = os.path.join(os.path.expanduser(output_dir), 'renders', 'sparse')
    im_dir   = os.path.join(meta_dir, modality, str(input_size))
    return (os.path.exists(im_dir), os.path.normpath(im_dir))


def write_dataset(sparse_dir, output_dir, modality, input_size, n_images, **_):
    sparse_dir = os.path.expanduser(sparse_dir)
    data_dir   = os.path.join(sparse_dir, 'sparse_database', f'database_{modality}')
    info_path  = os.path.join(data_dir, 'datasetInfoClean.csv')

    meta_dir = os.path.join(os.path.expanduser(output_dir), 'renders', 'sparse')
    im_dir   = os.path.join(meta_dir, modality, str(input_size))
    os.makedirs(im_dir, exist_ok=True)

    rd_it        = tuple(util.take(n_images, csv.iter_file(parse_line, info_path)))
    prog_printer = ft.partial(print_progress, len(rd_it))
    im_transform = make_im_transform(input_size)
    with open(os.path.join(meta_dir, 'meta.bin'), 'wb') as meta_f:
        prog_printer((0, None))
        util.dorun(
            map(prog_printer, enumerate(
                map(ft.partial(save_im_metadata, meta_f),
                    map(ft.partial(save_transformed_im, im_transform, modality, data_dir, im_dir),
                        rd_it)), 1)))
        pickle.dump('EOF', meta_f)

