#!/usr/bin/env python3

import argparse
import sys

from safe_gpu import safe_gpu

import src.deit_vis_loc.data  as data
import src.deit_vis_loc.model as model
import src.deit_vis_loc.util  as util


def parse_args(args_it):
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
    parser.add_argument('-s', '--dataset_size', type=int, default=None,
            required=False, help='When set it slices input datasets to n items')
    return vars(parser.parse_args(args_it))


if __name__ == "__main__":
    gpu_owner     = safe_gpu.GPUOwner()
    args          = parse_args(sys.argv[1:])
    train_params  = data.read_train_params(args['train_params'])
    segments_meta = data.read_segments_metadata(args, train_params['yaw_tolerance_deg'])
    iter_queries  = lambda f: data.read_query_imgs(args['segments_dataset'], f)
    query_images  = {
        'train': list(util.take(args['dataset_size'], iter_queries('train.txt'))),
        'val'  : list(util.take(args['dataset_size'], iter_queries('val.txt'))),
    }
    model.train(query_images, segments_meta, train_params, args['output'])

