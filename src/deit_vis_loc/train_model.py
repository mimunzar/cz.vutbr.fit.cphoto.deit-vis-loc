#!/usr/bin/env python3

import argparse
import json
<<<<<<< HEAD
=======
import operator as op
import os
>>>>>>> cb283e9 ([WIP] Computes Angle Difference)
import sys

import src.deit_vis_loc.model as model
import src.deit_vis_loc.utils as utils


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Allows to train DeiT transformers for visual localization.')
    parser.add_argument('-q', '--query_segments',
            required=True, help='The path to file containing queries with associated segments')
    parser.add_argument('-s', '--segments_dataset',
            required=True, help='The path to directory containing dataset of rendered segments')
    parser.add_argument('-t', '--train_params',
            required=True, help='The path to file which contains definition of train params')
    parser.add_argument('-o', '--output_dir',
            required=True, help='The path to directory where models are saved')
    return vars(parser.parse_args(list_of_args))


def parse_train_params(train_params):
    is_non_empty_str = lambda s: s.strip()
    is_positive      = lambda n: 0 < n
    is_positive_int  = lambda n: isinstance(n, int) and is_positive(n)
    checker = utils.make_checker({
        'batch_size'       : utils.make_validator('batch_size must be a positive int', is_positive_int),
        'deit_model'       : utils.make_validator('deit_model must be a non-empty string', is_non_empty_str),
        'max_epochs'       : utils.make_validator('max_epochs must be a positive int', is_positive_int),
        'triplet_margin'   : utils.make_validator('triplet_margin must be positive', is_positive),
        'learning_rate'    : utils.make_validator('learning_rate must be positive', is_positive),
        'stopping_patience': utils.make_validator('stopping_patience must be positive', is_positive),
    })
    if not checker(train_params):
        return train_params
    print('Invalid model params ({})'.format(', '.join(checker(train_params))), file=sys.stderr)
    sys.exit(1)


def positive_negative_segments(query):
    def has_similar_view_as_query(segment):
        yaw_angle = op.itemgetter('camera_orientation', 'yaw')
        return utils.angle_diff(yaw_angle(query), yaw_angle(segment)) <= ma.radians(30) + 1.e4

    segment_names = lambda list_segments: set(s['name'] for s in list_segments)
    p_seg, n_seg  = utils.partition_by(has_similar_view_as_query, query['segments'])
    return {'positive': segment_names(p_seg), 'negative': segment_names(n_seg)}


def parse_query_segments(query_records, dataset_dpath):
    to_query_path    = lambda s: os.path.join(dataset_dpath, 'query_original_result', s) + '.jpg'
    to_segment_path  = lambda s: os.path.join(dataset_dpath, 'database_segments', s) + '.png'
    map_segment_path = lambda s: {k: {to_segment_path(s) for s in v} for k, v in s.items()}

    pos_neg = ((k, positive_negative_segments(v)) for k, v in query_records.items())
    return {to_query_path(k): map_segment_path(v) for k, v in pos_neg}


def gen_query_imgs(dataset_dpath, name):
    queries_dpath = os.path.join(dataset_dpath, 'query_original_result')
    dataset_fpath = os.path.join(queries_dpath, name)
    return (os.path.join(queries_dpath, l.strip()) for l in open(dataset_fpath))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    with open(args['train_params']) as f:
        train_params = parse_train_params(json.load(f))
    with open(args['query_segments']) as f:
        query_segments = parse_query_segments(json.load(f), args['segments_dataset'])

    query_images   = {
        'train': list(gen_query_imgs(args['segments_dataset'], 'train.txt')),
        'val'  : list(gen_query_imgs(args['segments_dataset'], 'val.txt')),
    }
    # model.train(query_images, dataset_info, model_params, args['save_dir'])

