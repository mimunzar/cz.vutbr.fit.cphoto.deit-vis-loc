#!/usr/bin/env python3

import argparse
import json
import math as ma
import os
import sys

import src.deit_vis_loc.model as model
import src.deit_vis_loc.utils as utils


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Allows to train DeiT transformers for visual localization.')
    parser.add_argument('-r', '--rendered_segments',
            required=True, help='The path to file containing queries with associated segments')
    parser.add_argument('-s', '--segments_dataset',
            required=True, help='The path to directory containing dataset of rendered segments')
    parser.add_argument('-m', '--model_params',
            required=True, help='The path to file which contains definition of model params')
    parser.add_argument('-o', '--output_dir',
            required=True, help='The path to directory where models are saved')
    return vars(parser.parse_args(list_of_args))


def parse_model_params(model_params):
    is_non_empty_str = lambda s: s.strip()
    is_positive      = lambda n: 0 < n
    is_int           = lambda n: isinstance(n, int)
    is_positive_int  = lambda n: is_int(n) and is_positive(n)
    checker = utils.make_checker({
        'batch_size'        : utils.make_validator('batch_size must be a positive int', is_positive_int),
        'deit_model'        : utils.make_validator('deit_model must be a non-empty string', is_non_empty_str),
        'max_epochs'        : utils.make_validator('max_epochs must be a positive int', is_positive_int),
        'triplet_margin'    : utils.make_validator('triplet_margin must be positive', is_positive),
        'learning_rate'     : utils.make_validator('learning_rate must be positive', is_positive),
        'stopping_patience' : utils.make_validator('stopping_patience must be positive', is_positive),
        "yaw_difference_deg": utils.make_validator('yaw_difference_deg must be an int', is_int),
    })
    if not checker(model_params):
        return model_params
    print('Invalid model params ({})'.format(', '.join(checker(model_params))), file=sys.stderr)
    sys.exit(1)


def split_by_yaw_angle(query, max_yaw_diff):
    def has_similar_yaw_angle(segment, min_delta=1e-4):
        yaw_angle  = lambda d: d['camera_orientation']['yaw']
        difference = utils.circle_difference_rad(yaw_angle(query), yaw_angle(segment))
        return difference <= ma.radians(max_yaw_diff) + min_delta

    segment_names = lambda list_segments: set(s['name'] for s in list_segments)
    p_seg, n_seg  = utils.partition_by(has_similar_yaw_angle, query['segments'])
    return {'positive': segment_names(p_seg), 'negative': segment_names(n_seg)}


def parse_rendered_segments(rendered_segments, dataset_dpath, model_params):
    to_query_path    = lambda s: os.path.join(dataset_dpath, 'query_original_result', s) + '.jpg'
    to_segment_path  = lambda s: os.path.join(dataset_dpath, 'database_segments', s) + '.png'
    map_segment_path = lambda s: {k: {to_segment_path(s) for s in v} for k, v in s.items()}

    max_yaw_diff     = model_params['yaw_difference_deg']
    pos_neg_segments = ((k, split_by_yaw_angle(v, max_yaw_diff)) for k, v in rendered_segments.items())
    return {to_query_path(k): map_segment_path(v) for k, v in pos_neg_segments}


def gen_query_imgs(dataset_dpath, name):
    queries_dpath = os.path.join(dataset_dpath, 'query_original_result')
    dataset_fpath = os.path.join(queries_dpath, name)
    return (os.path.join(queries_dpath, l.strip()) for l in open(dataset_fpath))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    with open(args['model_params']) as f:
        model_params = parse_model_params(json.load(f))
    with open(args['rendered_segments']) as f:
        rendered_segments = parse_rendered_segments(json.load(f), args['segments_dataset'], model_params)

    query_images = {
        'train': list(gen_query_imgs(args['segments_dataset'], 'train.txt')),
        'val'  : list(gen_query_imgs(args['segments_dataset'], 'val.txt')),
    }
    model.train(query_images, rendered_segments, model_params, args['output_dir'])

