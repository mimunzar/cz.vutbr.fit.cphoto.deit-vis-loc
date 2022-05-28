#!/usr/bin/env python3

import argparse
import json
import random
import sys

import torch
import torch.cuda
import torch.distributed
import torch.multiprocessing.spawn
import torch.nn.parallel
import torch.optim

import src.deit_vis_loc.data.loader as loader
import src.deit_vis_loc.training.config as config
import src.deit_vis_loc.training.locate as locate
import src.deit_vis_loc.training.model as model
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.util as util


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',    help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',    help='The number of images in input datasets',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--params',      help='The path to file containing training parameters',
            required=True, metavar='FILE')
    parser.add_argument('--device',      help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args     = parse_args(sys.argv[1:])
    data_dir = args['data_dir']
    device   = args['device']
    n_images = args['n_images']

    with open(args['params'], 'r') as f:
        params = config.parse(json.load(f))

    net   = model.load(params['deit_model']).to(device)
    model = {
        'net'   : net,
        'device': device,
        'net'   : net,
        'optim' : torch.optim.SGD(net.parameters(), params['lr'], momentum=0.9)
    }
    vim_it = random.sample(tuple(
        loader.iter_queries(data_dir, params['input_size'], 'val')), k=n_images)
    tim_it = random.sample(tuple(
        loader.iter_queries(data_dir, params['input_size'], 'train')), k=n_images)
    rd_it  = loader.iter_pretraining_renders(data_dir, params['input_size'], 'segments')

    print(log.msg('Started training\n'))
    util.dorun(
        util.take(params['max_epochs'],
            locate.iter_training(model, params, vim_it, tim_it, rd_it)))
    print(log.msg('Finished training', prefix='\n'))

