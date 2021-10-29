#!/usr/bin/env python3


import itertools


def partition(n, iterable):
    i = iter(iterable)
    return itertools.takewhile(lambda l: len(l) == n,
            (list(itertools.islice(i, n)) for _ in itertools.repeat(None)))


def to_segment_path(query_path):
    return query_path \
        .replace('query_original_result', 'query_segments_result') \
        .replace('.jpg', '.png')

