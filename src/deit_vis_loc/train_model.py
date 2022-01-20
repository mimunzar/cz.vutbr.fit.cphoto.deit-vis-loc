#!/usr/bin/env python3

import argparse
import json
import sys

import src.deit_vis_loc.data  as data
import src.deit_vis_loc.model as model
import src.deit_vis_loc.util as util


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Allows to train DeiT transformers for visual localization.')
    parser.add_argument('-m', '--segments_meta',
            required=True, help='The path to file containing segments metadata')
    parser.add_argument('-d', '--segments_dataset',
            required=True, help='The path to directory containing dataset of rendered segments')
    parser.add_argument('-p', '--train_params',
            required=True, help='The path to file containing training parameters')
    parser.add_argument('-o', '--output',
            required=True, help='The path to directory where results are saved')
    return vars(parser.parse_args(list_of_args))


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


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    with open(args['train_params']) as f:
        train_params = parse_train_params(json.load(f))
    segments_meta = data.read_segments_metadata(args, train_params['yaw_tolerance_deg'])
    query_images  = {
        'train': data.read_query_imgs(args['segments_dataset'], 'train.txt'),
        'val'  : data.read_query_imgs(args['segments_dataset'], 'val.txt'),
    }
    model.train(query_images, segments_meta, train_params, args['output'])

