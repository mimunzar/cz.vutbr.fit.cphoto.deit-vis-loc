#!/usr/bin/env python3

import argparse
import json
import sys

import torch.hub
import torch.optim
import torch.cuda
from safe_gpu import safe_gpu

import src.deit_vis_loc.data     as data
import src.deit_vis_loc.training as training
import src.deit_vis_loc.util     as util


def parse_args(args_it):
    parser = argparse.ArgumentParser(
            description='Allows to train DeiT transformers for visual localization.')
    parser.add_argument('-q', '--queries_meta',
            required=True, help='The path to file containing metadata for query images')
    parser.add_argument('-d', '--segments_dataset',
            required=True, help='The path to directory containing dataset of rendered segments')
    parser.add_argument('-p', '--train_params',
            required=True, help='The path to file containing training parameters')
    parser.add_argument('-o', '--output',
            required=True, help='The path to directory where results are saved')
    parser.add_argument('-s', '--dataset_size', type=int, default=None,
            required=False, help='When set it slices input datasets to n items')
    return vars(parser.parse_args(args_it))


def device_name(device):
    return 'cpu' if 'cpu' == device else torch.cuda.get_device_name(device)


if __name__ == "__main__":
    gpu_owner    = safe_gpu.GPUOwner()
    args         = parse_args(sys.argv[1:])
    train_params = data.read_train_params(args['train_params'])
    queries_meta = data.read_queries_metadata(args['queries_meta'],
            args['segments_dataset'], train_params['yaw_tolerance_deg'])
    iter_queries = lambda f: data.read_query_imgs(args['segments_dataset'], f)
    query_images = {
        'train': set(util.take(args['dataset_size'], iter_queries('train.txt'))),
        'val'  : set(util.take(args['dataset_size'], iter_queries('val.txt'))),
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = torch.hub.load('facebookresearch/deit:main', train_params['deit_model'], pretrained=True)
    model.to(device)
    model_goods = {
        'model'     : model,
        'device'    : device,
        'optimizer' : torch.optim.Adam(model.parameters(), train_params['learning_rate']),
        'transform' : training.make_im_transform(device, train_params['input_size']),
    }

    util.log('Started training on "{}" with {}'.format(device_name(device), json.dumps(train_params, indent=4)))
    result = training.train(model_goods, train_params, queries_meta, query_images, args['output'])
    util.log('Training ended with the best model in epoch {}'.format(result['epoch']))

