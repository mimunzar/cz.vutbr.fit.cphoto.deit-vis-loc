#!/usr/bin/env python3

import collections as cl
import os


def parse_into(od, it):
    return {k: f(v) for (k, f), v in zip(od.items(), it)}


CAMERA_FIELDS = cl.OrderedDict({
    'yaw'  : lambda x: float(x),
    'pitch': lambda x: float(x),
    'roll' : lambda x: float(x),
})
INFO_FILE_FIELDS = cl.OrderedDict({
    'annotation' : lambda x: x,
    'camera'     : lambda x: parse_into(CAMERA_FIELDS, x.split()),
    'latitude'   : lambda x: float(x),
    'longitude'  : lambda x: float(x),
    'elevation'  : lambda x: float(x),
    'FOV'        : lambda x: float(x),
})

def parse_info_file(name, content_it):
    parsed = parse_into(INFO_FILE_FIELDS, content_it)
    return {
        'name'      : name,
        'latitude'  : parsed['latitude'],
        'longitude' : parsed['longitude'],
        'elevation' : parsed['elevation'],
        'yaw'       : parsed['camera']['yaw'],
        'pitch'     : parsed['camera']['pitch'],
        'roll'      : parsed['camera']['roll'],
        'FOV'       : parsed['FOV'],
    }


def parse_im_dir(dpath):
    _, name = os.path.split(dpath)
    ipath   = os.path.join(dpath, 'info.txt')
    try:
        return parse_info_file(name, (l.strip() for l in open(ipath)))
    except Exception as ex:
        raise ValueError(f'Failed to parse {ipath} ({ex})')


def parse(dpath):
    return map(parse_im_dir, (r for r, _, f in os.walk(dpath) if 'info.txt' in f))

