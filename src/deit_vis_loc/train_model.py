#!/usr/bin/env python3

import argparse
import os
import sys

import torch.cuda
import torch.distributed
import torch.hub
import torch.multiprocessing.spawn
import torch.nn.parallel
import torch.optim
import torch.nn
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
    parser.add_argument('--train-params', help='The path to file containing training parameters',
            required=True, metavar='FILE')
    parser.add_argument('--output-dir',   help='The path to directory where results are saved',
            required=True, metavar='DIR')
    parser.add_argument('--dataset-size', help='The number of images in input datasets',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--device',       help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--workers',      help='The number of workers to spawn',
            required=False, type=int, default=1, metavar='NUM')
    parser.add_argument('--sge',          help='When set it initializes training for SGE server',
            required=False, action='store_const', const=True, default=False)
    return vars(parser.parse_args(args_it))


def device_name(device):
    return torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU'


def comm_backend(device):
    return 'nccl' if device.startswith('cuda') else 'gloo'


def worker(pid, nprocs, model_goods, train_params, queries_meta, query_images, output_dir):
    torch.distributed.init_process_group(comm_backend(model_goods['device']), rank=pid, world_size=nprocs)
    util.log(f'Started training on "{device_name(model_goods["device"])}"', end='\n\n')
    result = training.train(model_goods, train_params, queries_meta, query_images, output_dir)
    util.log(f'Training ended with the best model in epoch {result["epoch"]}', start='\n')
    return 0


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args['sge']:
        gpu_owner = safe_gpu.GPUOwner(nb_gpus=args['workers'])
    if args['device'].startswith('cuda'):
        assert args['workers'] <= torch.cuda.device_count(), 'Not enough GPUs'

    train_params = data.read_train_params(args['train_params'])
    queries_meta = data.read_queries_metadata(args['metafile'],
            args['dataset_dir'], train_params['yaw_tolerance_deg'])
    iter_queries = lambda f: data.read_query_imgs(args['dataset_dir'], f)
    query_images = {
        'train': set(util.take(args['dataset_size'], iter_queries('train.txt'))),
        'val'  : set(util.take(args['dataset_size'], iter_queries('val.txt'))),
    }

    model = torch.hub.load('facebookresearch/deit:main', train_params['deit_model'], pretrained=True)
    model.to(args['device'])
    model_goods = {
        'model'     : model,
        'device'    : args['device'],
        'optimizer' : torch.optim.Adam(model.parameters(), train_params['learning_rate']),
    }

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.multiprocessing.spawn(worker, nprocs=args['workers'], args=(
        args['workers'], model_goods, train_params, queries_meta, query_images, args['output_dir']))
    sys.exit(0)

