#!/usr/bin/env python3

import argparse
import functools as fp
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


def allocate_model_for_process_on_gpu(pid, model):
    device = f'cuda:{pid}'
    torch.cuda.set_device(pid)
    model.cuda(device)
    model  = torch.nn.parallel.DistributedDataParallel(model, device_ids=[pid])
    return (model, device)


def allocate_model_for_process_on_cpu(model):
    device = 'cpu'
    model  = torch.nn.parallel.DistributedDataParallel(model, device_ids=None)
    return (model, device)


def allocate_model_for_process(pid, model, device):
    if device.startswith('cuda'):
        return allocate_model_for_process_on_gpu(pid, model)
    return allocate_model_for_process_on_cpu(model)


def init_process(pid, nprocs, device):
    backend = 'nccl' if device.startswith('cuda') else 'gloo'
    torch.distributed.init_process_group(backend, rank=pid, world_size=nprocs)


def device_name(device):
    if device.startswith('cuda'):
        return torch.cuda.get_device_name(device)
    return 'CPU'


def training_process(model, device, train_params, queries_meta, val_queries_it, output_dir, pid, nprocs, train_partitions):
    init_process(pid, nprocs, device)
    model, device = allocate_model_for_process(pid, model, device)
    optimizer     = torch.optim.Adam(model.parameters(), train_params['learning_rate'])
    model_goods   = {'model': model, 'device': device, 'optimizer': optimizer}
    query_images  = {'train': util.nth(pid, train_partitions), 'val': val_queries_it}

    util.log(f'Started training on "{device_name(device)}"', end='\n\n')
    result = training.train(pid, model_goods, train_params, queries_meta, query_images, output_dir)
    util.log(f'Training ended with the best model in epoch {result["epoch"]}', start='\n')
    return result


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args['sge']:
        gpu_owner = safe_gpu.GPUOwner(nb_gpus=args['workers'])
    if args['device'].startswith('cuda'):
        assert args['workers'] <= torch.cuda.device_count(), 'Not enough GPUs'

    train_params = data.read_train_params(args['train_params'])
    queries_meta = data.read_queries_metadata(args['metafile'], args['dataset_dir'], train_params['yaw_tolerance_deg'])
    iter_queries = lambda f: data.read_query_imgs(args['dataset_dir'], f)

    model            = torch.hub.load('facebookresearch/deit:main', train_params['deit_model'], pretrained=True)
    val_queries_it   = set(util.take(args['dataset_size'], iter_queries('val.txt')))
    train_queries_it = set(util.take(args['dataset_size'], iter_queries('train.txt')))
    train_partitions = tuple(map(tuple, util.partition(len(train_queries_it)//args['workers'], train_queries_it)))
    train_process     = fp.partial(training_process,
            model, args['device'], train_params, queries_meta, val_queries_it, args['output_dir'])

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.multiprocessing.spawn(train_process, nprocs=args['workers'], args=(args['workers'], train_partitions))

