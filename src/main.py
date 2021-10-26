#!/usr/bin/env python3

import os


DATASET_PATH = '.git/datasets/GeoPose3K_v2/'


def read_paths_for_queries(dpath, name):
    queries_dpath = os.path.join(dpath, 'query_original_result')
    spec_paths    = os.path.join(queries_dpath, name)
    if os.path.exists(spec_paths):
        with open(spec_paths) as f:
            return [os.path.join(queries_dpath, line.strip()) for line in f]
    raise FileNotFoundError(
            'Failed to read queries paths ({} not found in {})'.format(name, queries_dpath))


queries_paths = read_paths_for_queries(DATASET_PATH, 'train.txt')

