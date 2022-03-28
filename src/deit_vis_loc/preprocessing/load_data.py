#!/usr/bin/env python3

import itertools as it
import os
import pickle


def iter_im_data(dataset_dpath, name):
    dpath = os.path.join(dataset_dpath, '.preprocessed')
    with open(os.path.join(dpath, name), 'rb') as f:
        not_eof  = lambda x: 'EOF' != x
        unpickle = lambda _: pickle.load(f)
        yield from it.takewhile(not_eof, map(unpickle, it.repeat(None)))

