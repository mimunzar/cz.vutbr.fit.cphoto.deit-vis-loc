#!/usr/bin/env python3

import argparse
import sys
import torch.hub
import torch.optim

import src.deit_vis_loc.preprocessing.load_data as load_data
import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.training.model as model


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', help='The path to dataset of rendered segments',
            required=True, metavar='DIR')
    parser.add_argument('--n-images',    help='The amount of images to sample from the Dataset',
            required=True, type=int, metavar='INT')
    parser.add_argument('--n-epochs',    help='The number of epochs to iterate a model training',
            required=True, type=int, metavar='INT')
    parser.add_argument('--device',      help='The device to use',
            required=False, choices=['cpu', 'cuda'], default='cuda')
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args   = parse_args(sys.argv[1:])
    images =  {
        'train' : util.take(args['n_images'], load_data.iter_im_data(args['dataset_dir'], 'train.bin')),
        'val'   : util.take(args['n_images'], load_data.iter_im_data(args['dataset_dir'], 'val.bin')),
    }
    rd_it  = load_data.iter_im_data(args['dataset_dir'], 'renders.bin')
    params = {
        'deit_model' : 'deit_tiny_patch16_224',
        'input_size' : 224,
        'margin'     : 0.2,
        'n_triplets' : 10,
        'lr'         : 1e-3,
        'batch_size' : 8,
        'max_epochs' : args['n_epochs'],
        'patience'   : args['n_epochs'],
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
    result = model.train({
            'device' : args['device'],
            'net'    : net,
            'optim'  : optim,
        }, params, sys.stdout, images, rd_it)

