#!/usr/bin/env python3

import argparse
import json
import sys

import src.deit_vis_loc.model as model
import src.deit_vis_loc.utils as utils


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Allows to train DeiT transformers for visual localization.')
    parser.add_argument('-d', '--dataset_dir',
            required=True, help='GeoPose3K dataset directory path')
    parser.add_argument('-s', '--save_dir',
            required=True, help='Model save directory')
    parser.add_argument('-p', '--params',
            required=True, help='Model params difinition file path')
    return vars(parser.parse_args(list_of_args))


def parse_params(a_model_params):
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
    if not checker(a_model_params):
        return a_model_params
    print('Invalid model params ({})'.format(', '.join(checker(a_model_params))), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    with open(args['params']) as f:
        params = parse_params(json.load(f))
    model.train(args['dataset_dir'], args['save_dir'], params)

