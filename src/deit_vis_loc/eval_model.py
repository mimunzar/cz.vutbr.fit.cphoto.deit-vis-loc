#!/usr/bin/env python3

import argparse
import sys

import torch
import torch.cuda
from safe_gpu import safe_gpu

import src.deit_vis_loc.data     as data
import src.deit_vis_loc.training as training


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Allows to test trained models on visual localization.')
    parser.add_argument('-q', '--queries_meta',
            required=True, help='The path to file containing metadata for query images')
    parser.add_argument('-d', '--segments_dataset',
            required=True, help='The path to directory containing dataset of rendered segments')
    parser.add_argument('-m', '--model',
            required=True, help='The path to saved DeiT model')
    parser.add_argument('-i', '--input_size', type=int,
            required=True, help='The size (resolution) of input images.')
    parser.add_argument('-y', '--yaw_tolerance_deg', type=int,
            required=True, help='The yaw difference tolerance for queries and segments')
    return vars(parser.parse_args(list_of_args))


if __name__ == "__main__":
    gpu_owner    = safe_gpu.GPUOwner()
    args         = parse_args(sys.argv[1:])
    queries_meta = data.read_metafile(args['queries_meta'], args['segments_dataset'], args['yaw_tolerance_deg'])
    queries_it   = set(data.read_ims(args['segments_dataset'], 'test.txt'))

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model   = torch.load(args['model'])
    model.to(device)
    model_goods = {
        'model'     : model,
        'device'    : device,
        'transform' : training.make_im_transform(device, args['input_size']),
    }
    test_result  = training.eval(model_goods, queries_meta, queries_it)

