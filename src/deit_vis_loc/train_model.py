#!/usr/bin/env python3

import argparse
import functools as fp
import json
import os
import sys
from datetime import datetime

import torch
import torch.cuda
import torch.distributed
import torch.hub
import torch.multiprocessing.spawn
import torch.nn.parallel
import torch.optim
import torchvision.transforms
from PIL                 import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

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


def allocate_network_for_process_on_gpu(net, pid):
    device = f'cuda:{pid}'
    torch.cuda.set_device(pid)
    net.cuda(device)
    net  = torch.nn.parallel.DistributedDataParallel(net, device_ids=[pid])
    return (net, device)


def allocate_network_for_process_on_cpu(net):
    device = 'cpu'
    net  = torch.nn.parallel.DistributedDataParallel(net, device_ids=None)
    return (net, device)


def allocate_network_for_process(net, pid, device):
    if device.startswith('cuda'):
        return allocate_network_for_process_on_gpu(net, pid)
    return allocate_network_for_process_on_cpu(net)


def init_process(pid, nprocs, device):
    backend = 'nccl' if device.startswith('cuda') else 'gloo'
    torch.distributed.init_process_group(backend, rank=pid, world_size=nprocs)


def device_name(device):
    if device.startswith('cuda'):
        return torch.cuda.get_device_name(device)
    return 'CPU'


def make_im_transform(device, input_size):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size, interpolation=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return lambda fpath: to_tensor(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)


def make_save_net(net, fileprefix, output_dir):
    def save_net(epoch):
        epoch_str = str(epoch).zfill(3)
        torch.save(net, os.path.join(output_dir, f'{fileprefix}-{epoch_str}.torch'))
    return save_net


def training_process(train_params, queries_meta, pid, procinit):
    init_process(pid, procinit['nprocs'], procinit['device'])
    net, device  = allocate_network_for_process(procinit['net'], pid, procinit['device'])
    prefix, odir = util.pluck(['fileprefix', 'output_dir'], procinit)
    model        = {
        'net'       : net,
        'device'    : device,
        'optimizer' : torch.optim.Adam(net.parameters(), train_params['learning_rate']),
        'transform' : make_im_transform(device, train_params['input_size']),
        'save_net'  : make_save_net(net, prefix, odir) if 0 == pid else lambda *_: None,
    }
    query_images = {'train': util.nth(pid, procinit['train_parts_it']), 'val': procinit['vqueries_it']}

    with open(os.path.join(odir, f'{prefix}-{pid}.log'), 'w') as logfile:
        util.log(f'Started training on "{device_name(device)}"\n\n', file=logfile)
        result = training.train(model, train_params, logfile, queries_meta, query_images)
        util.log(f'Training ended with the best net in epoch {result["epoch"]}', start='\n', file=logfile)
        return result


def params_to_name(epoch_secs, train_params):
    timestr         = datetime.fromtimestamp(epoch_secs).strftime('%Y%m%dT%H%M%S')
    net, batch_size = util.pluck(['deit_model', 'batch_size'], train_params)
    return f'{timestr}-{net}-{batch_size}'


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args['sge']:
        from safe_gpu import safe_gpu
        gpu_owner = safe_gpu.GPUOwner(nb_gpus=args['workers'])
    if args['device'].startswith('cuda'):
        assert args['workers'] <= torch.cuda.device_count(), 'Not enough GPUs'

    train_params = data.read_train_params(args['train_params'])
    queries_meta = data.read_queries_metadata(args['metafile'], args['dataset_dir'], train_params['yaw_tolerance_deg'])
    worker       = fp.partial(training_process, train_params, queries_meta)

    iter_queries = lambda f: data.read_query_imgs(args['dataset_dir'], f)
    tqueries_it  = set(util.take(args['dataset_size'], iter_queries('train.txt')))
    fileprefix   = params_to_name(util.epoch_secs(), train_params)
    procinit     = {
        'nprocs'    : args['workers'],
        'output_dir': args['output_dir'],
        'fileprefix': fileprefix,

        'net'            : torch.hub.load('facebookresearch/deit:main', train_params['deit_model'], pretrained=True),
        'device'         : args['device'],
        'vqueries_it'    : set(util.take(args['dataset_size'], iter_queries('val.txt'))),
        'train_parts_it' : tuple(util.partition(len(tqueries_it)//args['workers'], tqueries_it))
    }

    os.makedirs(args['output_dir'], exist_ok=True)
    with open(os.path.join(args['output_dir'], f'{fileprefix}.json'), 'w') as f:
        json.dump(train_params, f, indent=4)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.multiprocessing.spawn(worker, nprocs=procinit['nprocs'], args=(procinit,))

