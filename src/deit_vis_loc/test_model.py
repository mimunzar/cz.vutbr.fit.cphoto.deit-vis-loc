#!/usr/bin/env python3

import argparse
import sys

import src.deit_vis_loc.data  as data
import src.deit_vis_loc.model as model


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Allows to test trained models on visual localization.')
    parser.add_argument('-m', '--segments_meta',
            required=True, help='The path to file containing segments metadata')
    parser.add_argument('-d', '--segments_dataset',
            required=True, help='The path to directory containing dataset of rendered segments')
    parser.add_argument('-i', '--model',
            required=True, help='The path to saved DeiT model')
    parser.add_argument('-y', '--yaw_tolerance_deg', type=int,
            required=True, help='The yaw difference tolerance for queries and segments')
    return vars(parser.parse_args(list_of_args))


if __name__ == "__main__":
    args               = parse_args(sys.argv[1:])
    rendered_segments  = data.read_segments_metadata(args, args['yaw_tolerance_deg'])
    list_of_query_imgs = list(data.read_query_imgs(args['segments_dataset'], 'test.txt'))
    test_result        = model.test(list_of_query_imgs, rendered_segments, args['model'])

