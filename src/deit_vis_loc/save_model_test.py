#!/usr/bin/env python3

import argparse
import pickle
import sys
import os

import torch

import src.deit_vis_loc.data     as data
import src.deit_vis_loc.training as training
import src.deit_vis_loc.util     as util


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Allows to test trained models on visual localization.')
    parser.add_argument('--metafile',     help='The path to file containing image metadata',
            required=True, metavar='FILE')
    parser.add_argument('--dataset-dir',  help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--dataset-size', help='The number of images in input datasets',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--output-dir',   help='The path to directory where results are saved',
            required=True, metavar='DIR')
    parser.add_argument('-m', '--model',  help='The path to saved model to evaluate',
            required=True, metavar='FILE')
    parser.add_argument('--device',       help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(list_of_args))


def modelpath_to_fileprefix(model_fpath):
    return os.path.basename('-'.join(model_fpath.split('-')[:-1]))


if __name__ == "__main__":
    args         = parse_args(sys.argv[1:])
    fileprefix   = modelpath_to_fileprefix(args['model'])
    train_params = data.read_train_params(os.path.join(os.path.dirname(args['model']), f'{fileprefix}.json'))
    meta         = data.read_metafile(args['metafile'], args['dataset_dir'], train_params['yaw_tolerance_deg'])
    test_im_it   = set(util.take(args['dataset_size'], data.read_ims(args['dataset_dir'], 'test.txt')))
    model        = {'device': args['device'], 'net': torch.load(args['model'], map_location=args['device'])}

    with open(os.path.join(args['output_dir'], f'{fileprefix}.bin'), 'wb') as f:
        pickle.dump(len(test_im_it), f)
        for im_score in training.test(model, train_params, meta, test_im_it):
            pickle.dump(im_score, f)

