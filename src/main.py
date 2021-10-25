#!/usr/bin/env python3

import os


DATASET_PATH = '.git/datasets/GeoPose3K_v2/'


def find_files(root, name):
    return [os.path.join(dpath, name)
        for dpath, _, filenames in os.walk(root) if name in filenames]


def read_queries_paths(dpath):
    queries_dpath = os.path.join(dpath, 'query_original_result')
    spec_paths    = find_files(queries_dpath, 'train.txt')
    if spec_paths:
        with open(spec_paths[0]) as f:
            return [os.path.join(queries_dpath, line.strip()) for line in f]
    raise ValueError('Failed to read queries paths (train.txt not found in {})'.format(queries_dpath))


queries_paths = read_queries_paths(DATASET_PATH)


