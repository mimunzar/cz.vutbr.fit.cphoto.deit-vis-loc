#!/usr/bin/env python3

import argparse
import sys

import src.deit_vis_loc.data.im_geopose as im_geopose
import src.deit_vis_loc.data.renders_geopose as renders_geopose
import src.deit_vis_loc.data.renders_geopose_panoramas as renders_geopose_panoramas
import src.deit_vis_loc.libs.log as log


def parse_args(args_it):
    parser = argparse.ArgumentParser()
    parser.add_argument('--geopose-dir', help='The path to Geopose dataset',
            required=True, metavar='DIR')
    parser.add_argument('--sparse-dir',  help='The path to Sparse dataset',
            required=True, metavar='DIR')
    parser.add_argument('--output-dir',  help='The output directory',
            required=True, metavar='DIR')
    parser.add_argument('--dataset',     help='The type of output dataset',
            required=True, choices=['sparse', 'pretraining'])
    parser.add_argument('--modality',    help='The modality of images',
            required=True, choices=['segments', 'silhouettes', 'depth'])
    parser.add_argument('--n-images',    help='The number of images in dataset',
            required=False, type=int, default=None, metavar='NUM')
    parser.add_argument('--resolution',  help='The resolution of images',
            required=True, metavar='INT', type=int)
    return vars(parser.parse_args(args_it))


RENDER_MODULES = {
    'pretraining' : renders_geopose,
    'sparse'      : renders_geopose_panoramas,
}

if '__main__' == __name__:
    args    = parse_args(sys.argv[1:])
    dataset = args['dataset']

    if im_geopose.dataset_exists(**args):
        print(log.msg(f'Found im dataset at {args["output_dir"]}'))
    else:
        print(log.msg(f'Creating im dataset at {args["output_dir"]}\n'))
        im_geopose.write_dataset(**args)
        print()

    if RENDER_MODULES[dataset].dataset_exists(**args):
        print(log.msg(f'Found {dataset} renders at {args["output_dir"]}'))
    else:
        print(log.msg(f'Creating {dataset} renders at {args["output_dir"]}\n'))
        RENDER_MODULES[dataset].write_dataset(**args)

