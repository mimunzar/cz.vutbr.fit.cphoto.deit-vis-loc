#!/usr/bin/env python3

import argparse
import random
import sys
import torch.hub
import torch.optim

import src.deit_vis_loc.preprocessing.load_data as load_data
import src.deit_vis_loc.training as training


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',    help='The amount of images to sample from the Dataset',
            required=True, type=int, metavar='INT')
    parser.add_argument('--device',      help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    im_it  = load_data.iter_im_data(args['dataset_dir'], 'train.bin')
    rd_it  = load_data.iter_im_data(args['dataset_dir'], 'renders.bin')
    params = {
        'deit_model' : 'deit_tiny_patch16_224',
        'input_size' : 224,
        'margin'     : 0.2,
        'n_triplets' : 10,
        'lr'         : 1e-3,
        'batch_size' : 16,
        'positives'  : {
            'dist_m'            : 100,
            'dist_tolerance_m'  : 10,
            'yaw_deg'           : 15,
            'yaw_tolerance_deg' : 1,
        },
        'negatives'  : {
            'dist_m'           : 200,
            'dist_tolerance_m' : 10,
        }
    }

    net    = torch.hub.load('facebookresearch/deit:main', params['deit_model'], pretrained=True).to(args['device'])
    optim  = torch.optim.AdamW(net.parameters(), params['lr'])
    result = training.train_batch({'device': args['device'], 'net': net, 'optim': optim},
        params, sys.stdout, rd_it, 1, 1, random.sample(tuple(im_it), k=params['batch_size']))

