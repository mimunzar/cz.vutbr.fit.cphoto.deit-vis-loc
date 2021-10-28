#!/usr/bin/env python3


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

