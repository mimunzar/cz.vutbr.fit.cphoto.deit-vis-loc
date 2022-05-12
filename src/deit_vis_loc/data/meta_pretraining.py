#!/usr/bin/env python3

import collections as cl
import os

import src.deit_vis_loc.data.commons as commons


INFO_FILE_FIELDS = cl.OrderedDict({
    'id'        : lambda x: x,
    'query'     : lambda x: x,
    'latitude'  : lambda x: float(x),
    'longitude' : lambda x: float(x),
    'elevation' : lambda x: float(x),
    'yaw'       : lambda x: float(x),
    'pitch'     : lambda x: float(x),
    'roll'      : lambda x: float(x),
    'fov'       : lambda x: float(x),
})

def parse_line(csv_it):
    parsed = commons.parse_into(INFO_FILE_FIELDS, csv_it)
    return {
        'name'      : parsed['query'],
        'query'     : parsed['query'],
        'latitude'  : parsed['latitude'],
        'longitude' : parsed['longitude'],
        'elevation' : parsed['elevation'],
        'yaw'       : parsed['yaw'],
        'pitch'     : parsed['pitch'],
        'roll'      : parsed['roll'],
    }


def info_filepath(dataset_dir, mod):
    return os.path.join(dataset_dir,
            'sparse_queries', f'query_{mod}', 'datasetInfoClean.csv')


def parse(dataset_dir, mod):
    return commons.parse_csv_file(parse_line, info_filepath(dataset_dir, mod))

