#!/usr/bin/env python3

import src.deit_vis_loc.plot as plot


def test_ims_of_query_with_n_segments():
    segments = [
        {'name': 's3', 'distance': 3, 'is_positive': False},
        {'name': 's1', 'distance': 1, 'is_positive': True},
        {'name': 's2', 'distance': 2, 'is_positive': False},
    ]
    assert list(plot.ims_of_query_with_n_segments(42, 'foo', [])) == [
            ('foo', 'black', 0)]
    assert list(plot.ims_of_query_with_n_segments(0,  'foo', segments)) == [
            ('foo', 'black', 0)]
    assert list(plot.ims_of_query_with_n_segments(1,  'foo', segments)) == [
            ('foo', 'black', 0), ('s1', 'green', 1)]
    assert list(plot.ims_of_query_with_n_segments(5,  'foo', segments)) == [
            ('foo', 'black', 0), ('s1', 'green', 1), ('s2', 'red', 2), ('s3', 'red', 3)]


def test_index_of_first_positive():
    pass

