#!/usr/bin/env python3

import json
import os

import src.deit_vis_loc.libs.util as util


def read_metafile(fpath):
    with open(fpath) as f:
        return json.load(f)


def read_ims(dataset_dpath, name):
    fpath = os.path.join(dataset_dpath, 'query_original_result', name)
    return map(lambda l: '.'.join(l.strip().split('.')[:-1]), open(fpath))


def parse_train_params(train_params):
    is_nonempty_or_null = lambda x: (x is None) or x.strip()
    is_positive         = lambda n: 0 < n
    is_int              = lambda n: isinstance(n, int)
    is_positive_int     = lambda n: is_int(n) and is_positive(n)
    checker             = util.make_checker({
        'batch_size'        : util.make_validator('batch_size must be a positive int', is_positive_int),
        'deit_model'        : util.make_validator('deit_model must be a non-empty string', is_nonempty_or_null),
        'input_size'        : util.make_validator('input size (resolution) must be a positive int', is_positive_int),
        'max_epochs'        : util.make_validator('max_epochs must be a positive int', is_positive_int),
        'margin'            : util.make_validator('margin must be positive', is_positive),
        'im_datapoints'     : util.make_validator('im_datapoints must be positive int', is_positive_int),
        'learning_rate'     : util.make_validator('learning_rate must be positive', is_positive),
        'stopping_patience' : util.make_validator('stopping_patience must be positive', is_positive),

        'pos_dist_m'        : util.make_validator('pos_dist_m must be positive int', is_positive_int),
        'dist_tolerance_m'  : util.make_validator('dist_tolerance_m must be positive int', is_positive_int),
        'yaw_deg'           : util.make_validator('yaw_deg must be positive int', is_positive_int),
        'yaw_tolerance_deg' : util.make_validator('yaw_tolerance_deg must be positive int', is_positive_int),
    })
    if checker(train_params):
        raise ValueError('Invalid train params ({})'.format(', '.join(checker(train_params))))
    return train_params


def read_train_params(fpath):
    with open(fpath) as f:
        return parse_train_params(json.load(f))

