#!/usr/bin/env python3

import argparse
import pickle
import sys
import os

import torch

import src.deit_vis_loc.training.model  as model
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser(
            description='Allows to test trained models and save the results')
    parser.add_argument('--metafile',     help='The path to file containing image metadata',
            required=True, metavar='FILE')
    parser.add_argument('--dataset-dir',  help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--dataset-size', help='The number of images in input datasets',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--output-dir',   help='The path to directory where results are saved',
            required=True, metavar='DIR')
    parser.add_argument('--model',        help='The path to saved model to evaluate',
            required=True, metavar='FILE')
    parser.add_argument('--device',       help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


def modelpath_to_fileprefix(model_fpath):
    return os.path.basename('-'.join(model_fpath.split('-')[:-1]))


if __name__ == "__main__":
    args         = parse_args(sys.argv[1:])
    fileprefix   = modelpath_to_fileprefix(args['model'])
    train_params = data.read_train_params(os.path.join(os.path.dirname(args['model']), f'{fileprefix}.json'))
    meta         = data.read_metafile(args['metafile'])
    test_im_it   = set(util.take(args['dataset_size'], data.read_ims(args['dataset_dir'], 'test.txt')))
    model        = {'device': args['device'], 'net': torch.load(args['model'], map_location=args['device'])}
    formatter    = util.make_progress_formatter(bar_width=40, total=len(test_im_it))
    avg_ims_sec  = util.make_avg_ims_sec()

    with open(os.path.join(args['output_dir'], f'{fileprefix}.bin'), 'wb') as f:
        pickle.dump(len(test_im_it), f)
        print(f'{formatter("Test", 0, 0)}', end='\r', flush=True)
        for i, im_score in enumerate(model.test(model, train_params, meta, test_im_it), start=1):
            pickle.dump(im_score, f)
            print(f'{formatter("Test", i, avg_ims_sec(1))}', end='\r', flush=True)

