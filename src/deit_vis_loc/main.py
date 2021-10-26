#!/usr/bin/env python3

import os


DATASET_PATH = '.git/datasets/GeoPose3K_v2/'
BATCH_SIZE   = 100


def read_queries_paths(dataset_path, name):
    queries_dpath = os.path.join(dataset_path, 'query_original_result')
    dataset_fpath = os.path.join(queries_dpath, name)
    if os.path.exists(dataset_fpath):
        with open(dataset_fpath) as f:
            return [os.path.join(queries_dpath, line.strip()) for line in f]
    raise FileNotFoundError(
            'Failed to read queries paths ({} not found in {})'.format(name, queries_dpath))


def partition(n, coll):
    assert 0 < n
    if not coll:
        return []
    part = coll[:n]
    if len(part) == n:
        return [part] + partition(n, coll[n:])
    return [part]


def to_segment_path(query_path):
    return query_path \
        .replace('query_original_result', 'query_segments_result') \
        .replace('.jpg', '.png')


def gen_triplets(list_of_query_paths, fn_to_segment_path):
    result             = []
    set_of_query_paths = set(list_of_query_paths)
    for query_path in list_of_query_paths:
        pos_segment = fn_to_segment_path(query_path)
        for neg_path in set_of_query_paths - set([query_path]):
            result.append({
                'A': query_path,
                'P': pos_segment,
                'N': fn_to_segment_path(neg_path)
            })
    return result


if __name__ == "__main__":
    list_of_batches  = partition(BATCH_SIZE, read_queries_paths(DATASET_PATH, 'train.txt'))
    list_of_triplets = (gen_triplets(b, to_segment_path) for b in list_of_batches)

