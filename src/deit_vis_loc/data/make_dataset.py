#!/usr/bin/env python3

import argparse
import sys

import src.deit_vis_loc.data.query_geopose as query_geopose
import src.deit_vis_loc.data.renders_pretraining as renders_pretraining
import src.deit_vis_loc.data.renders_sparse as renders_sparse
import src.deit_vis_loc.libs.log as log


RENDER_MODULES = {
    'pretraining' : renders_pretraining,
    'sparse'      : renders_sparse,
}

def write_renders(args):
    dataset          = args['dataset']
    exists, data_dir = RENDER_MODULES[dataset].dataset_exists(**args)
    if exists:
        print(log.msg(f'Found {dataset} renders at {data_dir}'))
    else:
        print(log.msg(f'Creating {dataset} renders at {data_dir}\n'))
        RENDER_MODULES[dataset].write_dataset(**args)


def write_queries(args):
    exists, data_dir = query_geopose.dataset_exists(**args)
    if exists:
        print(log.msg(f'Found im data at {data_dir}'))
    else:
        print(log.msg(f'Creating im data at {data_dir}\n'))
        query_geopose.write_dataset(**args)
        print()


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--geopose-dir',help='The path to Geopose dataset',
            required=True, metavar='DIR')
    parser.add_argument('--sparse-dir', help='The path to Sparse dataset',
            required=True, metavar='DIR')
    parser.add_argument('--output-dir', help='The output directory',
            required=True, metavar='DIR')
    parser.add_argument('--dataset',  help='The type of output dataset',
            required=True, choices=['sparse', 'pretraining'])
    parser.add_argument('--modality', help='The modality of images',
            required=True, choices=['segments', 'silhouettes', 'depth'])
    parser.add_argument('--n-images', help='The number of images in dataset',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--input-size', help='The resolution of images',
            required=True, metavar='INT', type=int)
    parser.add_argument('--scale-by-fov', help='When set scales images by their FOV',
            required=False, action="store_true")
    return vars(parser.parse_args(args_it))


if '__main__' == __name__:
    args = parse_args(sys.argv[1:])
    write_queries(args)
    write_renders(args)
    sys.exit(0)

