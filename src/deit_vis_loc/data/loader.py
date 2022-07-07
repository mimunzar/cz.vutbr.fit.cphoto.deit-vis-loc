#!/usr/bin/env python3

import functools as ft
import itertools as it
import os
import pickle

import src.deit_vis_loc.libs.util as util


def iter_metafile(im_dir, meta_name):
    with open(os.path.join(im_dir, meta_name), 'rb') as f:
        not_eof  = lambda x: 'EOF' != x
        unpickle = ft.partial(pickle.load, f)
        yield from it.takewhile(not_eof, util.repeatedly(unpickle))


META_NAMES = {
    'TRAIN' : 'train.bin',
    'VAL'   : 'val.bin',
    'TEST'  : 'test.bin',
}

def assign_im_dir(im_dir, extension, meta):
    im_name = meta['name']
    return util.assoc(meta,
            ('path', os.path.join(im_dir, f'{im_name}.{extension}')))
    #^ We don't store an image path in image  metadata,  because  there  can  be
    # multiple versions of one image (e.g.  image stored in multiple input_sizes
    # for different models).


def iter_queries(member, data_dir, input_size, scale_by_fov, **_):
    query_dir    = os.path.join(os.path.expanduser(data_dir), 'queries')
    query_im_dir = os.path.join(query_dir,
            'data_fov' if scale_by_fov else 'data', str(input_size))
    if not os.path.exists(query_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {query_im_dir}')
    return map(ft.partial(assign_im_dir, query_im_dir, 'jpg'),
            iter_metafile(query_dir, META_NAMES[member.upper()]))
    #^
    #       queries
    #       ├── data
    #       │   └── <input_size>
    #       ├── data_fov
    #       │   └── <input_size>
    #       ├── test.bin
    #       ├── train.bin
    #       └── val.bin


RENDER_EXTENSION = {
    'segments'    : 'png',
    'silhouettes' : 'png',
    'depth'       : 'exr',
}

def iter_renders_pretraining(data_dir, input_size, modality, scale_by_fov, **_):
    data_dir      = os.path.expanduser(data_dir)
    render_dir    = os.path.join(data_dir, 'renders', 'pretraining')
    render_im_dir = os.path.join(render_dir,
            modality, 'data_fov' if scale_by_fov else 'data', str(input_size))
    if not os.path.exists(render_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {render_im_dir}')
    return map(ft.partial(assign_im_dir, render_im_dir, RENDER_EXTENSION[modality]),
            iter_metafile(render_dir, 'meta.bin'))
    #^
    #        renders/pretraining
    #        ├── meta.bin
    #        └── <modality>
    #            ├── data
    #            │   └── <input_size>
    #            └── data_fov
    #                └── <input_size>


def iter_renders_sparse(data_dir, input_size, modality, **_):
    data_dir      = os.path.expanduser(data_dir)
    render_dir    = os.path.join(data_dir, 'renders', 'sparse')
    render_im_dir = os.path.join(render_dir, modality, str(input_size))
    if not os.path.exists(render_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {render_im_dir}')
    return map(ft.partial(assign_im_dir, render_im_dir, RENDER_EXTENSION[modality]),
            iter_metafile(render_dir, 'meta.bin'))
    #^
    #       renders/sparse
    #       ├── meta.bin
    #       └── <modality>
    #           └── <input_size>

