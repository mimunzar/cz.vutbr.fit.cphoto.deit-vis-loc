#!/usr/bin/env python3

import argparse
import json
import sys

import src.deit_vis_loc.data  as data
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
        'yaw_tolerance_deg': utils.make_validator('yaw_tolerance_deg must be an int', is_int),
    })
    if not checker(model_params):
        return model_params
    print('Invalid model params ({})'.format(', '.join(checker(model_params))), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    with open(args['model_params']) as f:
        model_params  = parse_model_params(json.load(f))
    rendered_segments = data.read_rendered_segments(args, model_params['yaw_tolerance_deg'])
    query_images      = {
        'train': data.read_query_imgs(args['segments_dataset'], 'train.txt'),
        'val'  : data.read_query_imgs(args['segments_dataset'], 'val.txt'),
    }
    model.train(query_images, rendered_segments, model_params, args['output_dir'])

