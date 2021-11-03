#!/usr/bin/env python3

import argparse
import sys

import src.deit_vis_loc.model as model


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Allows to test trained models on visual localization.')
    parser.add_argument('-d', '--dataset_dir',
            required=True, help='GeoPose3K dataset directory path')
    parser.add_argument('-m', '--model',
            required=True, help='Path to a saved model')
    return vars(parser.parse_args(list_of_args))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    g = model.test(args['dataset_dir'], args['model'])

