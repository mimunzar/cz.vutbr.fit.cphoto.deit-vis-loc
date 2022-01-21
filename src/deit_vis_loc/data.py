#!/usr/bin/env python3

import itertools as it
import json
import math as ma
import os
import sys

import src.deit_vis_loc.util as util


def is_circle_diff_close(tolerance_rad, l_rad, r_rad):
    circle_diff_rad = util.circle_difference_rad(l_rad, r_rad)
    return circle_diff_rad <= tolerance_rad + 1e-4
    #^ Circle difference has to be lower than tolerance + precision


def split_segments_by_yaw(tolerance_rad, yaw_angle_rad, list_of_segments):
    yaw           = lambda s: s ['camera_orientation']['yaw']
    yaw_proximity = lambda s: is_circle_diff_close(tolerance_rad, yaw_angle_rad, yaw(s))
    return util.partition_by(yaw_proximity, list_of_segments)


def split_query_segments_by_yaw(query, yaw_tolerance_rad):
    yaw      = query['camera_orientation']['yaw']
    pos, neg = split_segments_by_yaw(yaw_tolerance_rad, yaw, query['segments'])
    names_of = lambda list_segments: set(s['name'] for s in list_segments)
    return {'positive': names_of(pos), 'negative': names_of(neg)}


def parse_segments_metadata(segments_meta, dataset_dpath, yaw_tolerance_rad):
    split_segments   = lambda k, v: (k, split_query_segments_by_yaw(v, yaw_tolerance_rad))
    pos_neg_segments = it.starmap(split_segments, segments_meta.items())

    to_query_path    = lambda s: os.path.join(dataset_dpath, 'query_original_result', s) + '.jpg'
    to_segment_path  = lambda s: os.path.join(dataset_dpath, 'database_segments', s) + '.png'
    map_segment_path = lambda s: {k: {to_segment_path(s) for s in v} for k, v in s.items()}
    return {to_query_path(k): map_segment_path(v) for k, v in pos_neg_segments}


def read_segments_metadata(args, yaw_tolerance_deg):
    tolerance_rad = ma.radians(yaw_tolerance_deg)
    with open(args['segments_meta']) as f:
        return parse_segments_metadata(json.load(f), args['segments_dataset'], tolerance_rad)


def read_query_imgs(dataset_dpath, name):
    queries_dpath = os.path.join(dataset_dpath, 'query_original_result')
    dataset_fpath = os.path.join(queries_dpath, name)
    return (os.path.join(queries_dpath, l.strip()) for l in open(dataset_fpath))


def parse_train_params(train_params):
    is_non_empty_str = lambda s: s.strip()
    is_positive      = lambda n: 0 < n
    is_int           = lambda n: isinstance(n, int)
    is_positive_int  = lambda n: is_int(n) and is_positive(n)
    checker = util.make_checker({
        'batch_size'        : util.make_validator('batch_size must be a positive int', is_positive_int),
        'deit_model'        : util.make_validator('deit_model must be a non-empty string', is_non_empty_str),
        'max_epochs'        : util.make_validator('max_epochs must be a positive int', is_positive_int),
        'triplet_margin'    : util.make_validator('triplet_margin must be positive', is_positive),
        'learning_rate'     : util.make_validator('learning_rate must be positive', is_positive),
        'stopping_patience' : util.make_validator('stopping_patience must be positive', is_positive),
        'yaw_tolerance_deg' : util.make_validator('yaw_tolerance_deg must be an int', is_int),
    })
    if not checker(train_params):
        return train_params
    print('Invalid model params ({})'.format(', '.join(checker(train_params))), file=sys.stderr)
    sys.exit(1)


def read_train_params(fpath):
    with open(fpath) as f:
        return parse_train_params(json.load(f))

