import argparse
import collections as cl
import itertools   as it
import json
import operator as op
import os
import re
import sys


def parse_args(list_of_args):
    parser = argparse.ArgumentParser(
            description='Associates queries from GeoPose3K dataset with their rendered segments')
    parser.add_argument('-g', '--geopose_dataset',
            required=True, help='The path to directory containing GeoPose3K dataset')
    parser.add_argument('-s', '--segments_dataset',
            required=True, help='The path to directory containing dataset of rendered segments')
    parser.add_argument('-o', '--output_file',
            required=True, help='The path for the output dataset description')
    return vars(parser.parse_args(list_of_args))


def parse_into(od, it):
    return {k: f(v) for (k, f), v in zip(od.items(), it)}


def geopose_img_metadata(fpath):
    camera_fields = cl.OrderedDict({
        'yaw'  : lambda x: float(x),
        'pitch': lambda x: float(x),
        'roll' : lambda x: float(x),
    })
    info_file_fields = cl.OrderedDict({
        'annotation'         : lambda x: x,
        'camera_orientation' : lambda x: parse_into(camera_fields, x.split()),
        'estimated_latitude' : lambda x: float(x),
        'estimated_longitude': lambda x: float(x),
        'estimated_elevation': lambda x: float(x),
        'estimated_FOV'      : lambda x: float(x),
    })
    return parse_into(info_file_fields, (l.strip() for l in open(fpath)))


def geopose_metadata(dpath):
    dpath   = os.path.expanduser(dpath)
    queries = (os.path.basename(r) for r, _, fs in os.walk(dpath) if 'info.txt' in fs)
    return {q: geopose_img_metadata(os.path.join(dpath, q, 'info.txt')) for q in queries}


def del_whitespace(s):
    return re.sub(r'\s+', '', s)


def segment_img_metadata(segment):
    return {
        'name'     : segment['segment'],
        'latitude' : segment['latitude'],
        'longitude': segment['longitude'],
        'elevation': segment['elevation'],
        'camera_orientation' : {
            'yaw'  : segment['rotation'],
            'pitch': 0,
            'roll' : 0,
        }
    }
    #^ In datasetInfoClean.csv the rotation fields coresponds to yaw angles
    # for geopose images and the pitch and roll are 0.


def segment_metadata(dpath):
    info_file_fpath  = os.path.join(os.path.expanduser(dpath), 'database_segments/datasetInfoClean.csv')
    info_file_fields = cl.OrderedDict({
        'segment'   : lambda x: x + '_segments',
        'query'     : lambda x: x,
        'latitude'  : lambda x: float(x),
        'longitude' : lambda x: float(x),
        'elevation' : lambda x: float(x),
        'rotation'  : lambda x: float(x),
        'yaw'       : lambda x: float(x),
        'pitch'     : lambda x: float(x),
        'roll'      : lambda x: float(x),
    })

    segments = (parse_into(info_file_fields, del_whitespace(l).split(',')) for l in open(info_file_fpath))
    by_query = it.groupby(sorted(segments, key=op.itemgetter('query')), op.itemgetter('query'))
    return {q: {'segments': [segment_img_metadata(s) for s in segs]} for q, segs in by_query}


def merge_by_query(dl, dr):
    assert sorted(dl.keys()) == sorted(dr.keys()), "Queries are expected to be the same"
    return {k: {**dl[k], **dr[k]} for k in dl.keys()}


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    with open(args['output_file'], 'w') as f:
        g = geopose_metadata(args['geopose_dataset'])
        s = segment_metadata(args['segments_dataset'])
        json.dump(merge_by_query(g, s), f, indent=4)

