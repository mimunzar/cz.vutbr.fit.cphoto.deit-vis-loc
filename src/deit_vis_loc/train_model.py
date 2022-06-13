#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.cuda
import torch.distributed
import torch.multiprocessing.spawn
import torch.nn.parallel
import torch.optim

import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.config as config
import src.deit_vis_loc.training.crosslocate as crosslocate
import src.deit_vis_loc.training.model as model
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util


RENDER_LOADERS = {
    'pretraining' : loader.iter_pretraining_renders,
    'sparse'      : loader.iter_sparse_renders,
}


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',    help='The path to the dataset',
            required=True, metavar='DIR')
    parser.add_argument('--dataset',     help='The type of the input dataset',
            required=True, choices=['sparse', 'pretraining'])
    parser.add_argument('--modality',    help='The modality of images',
            required=True, choices=['segments', 'silhouettes', 'depth'])
    parser.add_argument('--n-images',    help='The number of images',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--output-dir',  help='The output directory',
            required=True, metavar='DIR')
    parser.add_argument('--params',      help='The file path to training definition',
            required=True, metavar='FILE')
    parser.add_argument('--device',      help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


def device_name(device):
    return torch.cuda.get_device_name(device) if 'cuda' == device else 'CPU'


def params_to_fileprefix(epoch_secs, params):
    timestr         = datetime.fromtimestamp(epoch_secs).strftime('%Y%m%dT%H%M%S')
    net, batch_size = util.pluck(['deit_model', 'batch_size'], params)
    return f'{timestr}-{net}-{batch_size}'


if '__main__' == __name__:
    args     = parse_args(sys.argv[1:])
    data_dir = args['data_dir']
    device   = args['device']
    n_images = args['n_images']

    with open(args['params'], 'r') as f:
        params = config.parse(json.load(f))

    prefix  = params_to_fileprefix(util.epoch_secs(), params)
    out_dir = os.path.join(os.path.expanduser(args['output_dir']), prefix)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    net   = model.new(params['deit_model']).to(device)
    model = {
        'net'   : net,
        'device': device,
        'optim' : torch.optim.SGD(net.parameters(), params['lr'], momentum=0.9)
    }
    vim_it = util.take(n_images, loader.iter_queries(data_dir, params['input_size'], 'val'))
    tim_it = util.take(n_images, loader.iter_queries(data_dir, params['input_size'], 'train'))
    rd_it  = RENDER_LOADERS[args['dataset']](data_dir, params['input_size'], args['modality'])

    print(log.msg(f'Started training process "{os.getpid()}" on "{device_name(device)}"\n'))
    crosslocate.train(model, params, out_dir, vim_it, tim_it, rd_it)
    print(log.msg('Finished training', prefix='\n'))

