#!/usr/bin/env python3

import functools as ft
import itertools as it
import os
import pickle

import src.deit_vis_loc.libs.util as util


def iter_metafile(im_dir, meta_name):
    with open(os.path.join(im_dir, meta_name), 'rb') as f:
        not_eof  = lambda x: 'EOF' != x
        unpickle = lambda _: pickle.load(f)
        yield from it.takewhile(not_eof, map(unpickle, it.repeat(None)))


META_NAMES = {
    'TRAIN' : 'train.bin',
    'VAL'   : 'val.bin',
    'TEST'  : 'test.bin',
}

def assign_query_path(im_dir, meta):
    return util.assoc('path', os.path.join(im_dir, f'{meta["name"]}.jpg'), meta)


def iter_queries(data_dir, resolution, member):
    data_dir     = os.path.expanduser(data_dir)
    query_dir    = os.path.join(data_dir, 'queries')
    query_im_dir = os.path.join(query_dir, str(resolution))
    if not os.path.exists(query_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {query_im_dir}')
    return map(ft.partial(assign_query_path, query_im_dir),
            iter_metafile(query_dir, META_NAMES[member.upper()]))


def iter_pretraining_renders(data_dir, resolution, modality):
    data_dir      = os.path.expanduser(data_dir)
    render_dir    = os.path.join(data_dir, 'renders', 'pretraining', modality)
    render_im_dir = os.path.join(render_dir, str(resolution))
    if not os.path.exists(render_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {render_im_dir}')
    return map(ft.partial(assign_query_path, render_im_dir), iter_metafile(render_dir, 'meta.bin'))


def iter_sparse_renders(data_dir, resolution, modality):
    data_dir      = os.path.expanduser(data_dir)
    render_dir    = os.path.join(data_dir, 'renders', 'sparse', modality)
    render_im_dir = os.path.join(render_dir, str(resolution))
    if not os.path.exists(render_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {render_im_dir}')
    return map(ft.partial(assign_query_path, render_im_dir), iter_metafile(render_dir, 'meta.bin'))

