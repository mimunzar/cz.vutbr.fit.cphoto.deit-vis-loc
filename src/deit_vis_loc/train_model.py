#!/usr/bin/env python3

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime

import torch
import torch.cuda
import torch.distributed
import torch.multiprocessing.spawn
import torch.nn.parallel
import torch.optim

import src.deit_vis_loc.preprocessing.load_data as load_data
import src.deit_vis_loc.training.callbacks as callbacks
import src.deit_vis_loc.training.config as config
import src.deit_vis_loc.training.locate as locate
import src.deit_vis_loc.training.model as model
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',  help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',     help='The number of images in input datasets',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--train-params', help='The path to file containing training parameters',
            required=True, metavar='FILE')
    parser.add_argument('--output-dir',   help='The path to directory where results are saved',
            required=True, metavar='DIR')
    parser.add_argument('--device',       help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--workers',      help='The number of workers to spawn',
            required=False, type=int, default=1, metavar='NUM')
    parser.add_argument('--sge',          help='When set it initializes training for SGE server',
            required=False, action='store_const', const=True, default=False)
    return vars(parser.parse_args(args_it))


def allocate_network_for_process_on_gpu(net, pid):
    device = f'cuda:{pid}'
    with torch.cuda.device(device):
        net.cuda(device)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[pid])
        return (net, device)


def allocate_network_for_process_on_cpu(net):
    device = 'cpu'
    net    = torch.nn.parallel.DistributedDataParallel(net, device_ids=None)
    return (net, device)


def allocate_network_for_process(pid, net, device, **_):
    if 'cuda' == device:
        return allocate_network_for_process_on_gpu(net, pid)
    return allocate_network_for_process_on_cpu(net)


def init_process(pid, nprocs, device, **_):
    backend = 'nccl' if device.startswith('cuda') else 'gloo'
    torch.distributed.init_process_group(backend, rank=pid, world_size=nprocs)


def device_name(device, **_):
    return torch.cuda.get_device_name(device) if 'cuda' == device else 'CPU'


def training(pid, init):
    init_process(pid, **init)
    net, device  = allocate_network_for_process(pid, **init)
    with open(os.path.join(init['output_dir'], f'{init["prefix"]}-{pid}.log'), 'w') as logfile:
        log.log(f'Started training process "{os.getpid()}" on "{device_name(**init)}"', file=logfile)
        result = locate.train(logfile, init['params'], {
                'net'      : net,
                'device'   : device,
                'optim'    : torch.optim.SGD(net.parameters(), init['params']['lr'], momentum=0.9),
            }, {
                'train'   : util.nth(pid, init['im_batch_it']),
                'val'     : init['val_it'],
                'renders' : init['renders_it'],
            }, [
                callbacks.make_save_net(net, **init),
                callbacks.make_plot_batch_loss(**init),
                callbacks.make_plot_epoch_loss(**init),
                callbacks.make_plot_epoch_samples(**init),
            ] if 0 == pid else [])
        log.log(f'Training ended with the best epoch {result["epoch"]}', start='\n', file=logfile)
        return result


def params_to_fileprefix(epoch_secs, train_params):
    timestr         = datetime.fromtimestamp(epoch_secs).strftime('%Y%m%dT%H%M%S')
    net, batch_size = util.pluck(['deit_model', 'batch_size'], train_params)
    return f'{timestr}-{net}-{batch_size}'


def nvidia_cmd(query):
    return subprocess.check_output(['nvidia-smi', query, '--format=noheader,csv'], encoding='utf-8')


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args['sge'] and 'cuda' == args['device']:
        devices      = set(nvidia_cmd('--query-gpu=gpu_uuid').split('\n'))
        busy_devices = set(nvidia_cmd('--query-compute-apps=gpu_uuid').split('\n'))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices - busy_devices)

    if 'cuda' == args['device']:
        assert args['workers'] <= torch.cuda.device_count(), 'Not enough GPUs'

    with open(args['train_params'], 'r') as f:
        params = config.parse(json.load(f))

    prefix  = params_to_fileprefix(util.epoch_secs(), params)
    out_dir = os.path.join(args['output_dir'], prefix)
    im_it   = tuple(util.take(args['n_images'], load_data.iter_im_data(args['dataset_dir'], 'train.bin')))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{prefix}.json'), 'w') as f:
        json.dump(params, f, indent=4)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    torch.multiprocessing.spawn(training, nprocs=args['workers'], args=({
            'nprocs'      : args['workers'],
            'output_dir'  : out_dir,
            'prefix'      : prefix,
            'net'         : model.load(params['deit_model']),
            'device'      : args['device'],
            'params'      : params,
            'val_it'      : list(util.take(args['n_images'], load_data.iter_im_data(args['dataset_dir'], 'val.bin'))),
            'im_batch_it' : tuple(util.partition(len(im_it)//args['workers'], im_it)),
            'renders_it'  : tuple(load_data.iter_im_data(args['dataset_dir'], 'renders.bin')),
        },))

