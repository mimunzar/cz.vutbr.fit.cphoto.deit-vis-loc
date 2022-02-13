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
    parser = argparse.ArgumentParser()
    parser.add_argument('--metafile',     help='The path to file containing metadata for queries',
            required=True, metavar='FILE')
    parser.add_argument('--dataset-dir',  help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--dataset-size', help='When set it slices input datasets to n items',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--train-params', help='The path to file containing training parameters',
            required=True, metavar='FILE')
    parser.add_argument('--output-dir',   help='The path to directory where results are saved',
            required=True, metavar='DIR')
    parser.add_argument('--sge',          help='When set it initializes training for SGE server',
            required=False, action='store_const', const=True, default=False)
    return vars(parser.parse_args(args_it))


def device_name(device):
    return 'cpu' if 'cpu' == device else torch.cuda.get_device_name(device)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args['sge']:
        gpu_owner = safe_gpu.GPUOwner()
    train_params = data.read_train_params(args['train_params'])
    queries_meta = data.read_queries_metadata(args['metafile'],
            args['dataset_dir'], train_params['yaw_tolerance_deg'])
    iter_queries = lambda f: data.read_query_imgs(args['dataset_dir'], f)
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

    util.log(f'Started training on "{device_name(device)}" with {json.dumps(train_params, indent=4)}')
    result = training.train(model_goods, train_params, queries_meta, query_images, args['output_dir'])
    util.log(f'Training ended with the best model in epoch {result["epoch"]}', start='\n')

