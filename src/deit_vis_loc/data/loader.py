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

def assign_im_dir(im_dir, meta):
    return util.assoc(meta, ('path', os.path.join(im_dir, f'{meta["name"]}.jpg')))
    #^ We don't store an image path in image  metadata,  because  there  can  be
    # multiple versions of one image (e.g.  image stored in multiple input_sizes
    # for different models).


def iter_queries(data_dir, input_size, member):
    data_dir     = os.path.expanduser(data_dir)
    query_dir    = os.path.join(data_dir, 'queries')
    query_im_dir = os.path.join(query_dir, str(input_size))
    if not os.path.exists(query_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {query_im_dir}')
    return map(ft.partial(assign_im_dir, query_im_dir),
            iter_metafile(query_dir, META_NAMES[member.upper()]))
    #^ Queries are stored in the following directory  structure  which  contains
    # image metadata splitted into train, val and  test  datasets.   The  actual
    # images are stored per input_size in subdirectories  to  support  different
    # models.
    #
    #       queries/
    #       ├── <input_size1>
    #       ├── <input_size2>
    #       ├── test.bin
    #       ├── train.bin
    #       └── val.bin



def iter_pretraining_renders(data_dir, input_size, modality):
    data_dir      = os.path.expanduser(data_dir)
    render_dir    = os.path.join(data_dir, 'renders', 'pretraining', modality)
    render_im_dir = os.path.join(render_dir, str(input_size))
    if not os.path.exists(render_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {render_im_dir}')
    return map(ft.partial(assign_im_dir, render_im_dir), iter_metafile(render_dir, 'meta.bin'))
    #^ Renders are stored in the  following  directory  structure  splitted  per
    # dataset, modality and  input_size.   The  metadata  differ  for  different
    # datasets and modalities.  As with queries render  images  are  stored  per
    # input_size to support different models.
    #
    #       renders/
    #       ├── pretraining
    #       │   ├── <modality1>
    #       │   └── <modality2>
    #       │       ├── <input_size1>
    #       │       ├── <input_size2>
    #       │       └── meta.bin
    #       ├── sparse
    #       │   ├── <modality1>
    #       │   └── <modality2>
    #       │       ├── <input_size1>
    #       │       ├── <input_size2>
    #       │       └── meta.bin
    #       └── uniform
    #           └── ...


def iter_sparse_renders(data_dir, input_size, modality):
    data_dir      = os.path.expanduser(data_dir)
    render_dir    = os.path.join(data_dir, 'renders', 'sparse', modality)
    render_im_dir = os.path.join(render_dir, str(input_size))
    if not os.path.exists(render_im_dir):
        raise FileNotFoundError(f'Image directory doesnt exists {render_im_dir}')
    return map(ft.partial(assign_im_dir, render_im_dir), iter_metafile(render_dir, 'meta.bin'))

