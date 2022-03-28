#!/usr/bin/env python3

import collections as cl
import os
import re


def parse_into(od, it):
    return {k: f(v) for (k, f), v in zip(od.items(), it)}


INFO_FILE_FIELDS = cl.OrderedDict({
    'segment'   : lambda x: x,
    'query'     : lambda x: x,
    'latitude'  : lambda x: float(x),
    'longitude' : lambda x: float(x),
    'elevation' : lambda x: float(x),
    'rotation'  : lambda x: float(x),
    'yaw'       : lambda x: float(x),
    'pitch'     : lambda x: float(x),
    'roll'      : lambda x: float(x),
})

def parse_info_line(content_it):
    parsed = parse_into(INFO_FILE_FIELDS, content_it)
    return {
        'name'      : parsed['segment'],
        'query'     : parsed['query'],
        'latitude'  : parsed['latitude'],
        'longitude' : parsed['longitude'],
        'elevation' : parsed['elevation'],
        'yaw'       : parsed['rotation'],
        'pitch'     : 0,
        'roll'      : 0,
    }
    #^ In datasetInfoClean.csv the rotation fields coresponds to yaw angles
    # for geopose images and the pitch and roll are 0.


def parse(dpath):
    white_re = re.compile(r'\s+')
    inf_path = os.path.join(os.path.expanduser(dpath), 'database_segments/datasetInfoClean.csv')
    try:
        return map(parse_info_line, (white_re.sub('', l).split(',') for l in open(inf_path)))
    except Exception as ex:
        raise ValueError(f'Failed to parse {dpath} ({ex})')

