#!/usr/bin/env python3

import src.deit_vis_loc.main as main

from operator import itemgetter


def test_partition():
    assert main.partition(1, []) == []
    assert main.partition(1, [1, 2, 3]) == [[1], [2], [3]]
    assert main.partition(2, [1, 2, 3]) == [[1, 2], [3]]
    assert main.partition(3, [1, 2, 3]) == [[1, 2, 3]]
    assert main.partition(4, [1, 2, 3]) == [[1, 2, 3]]


def test_generate_triplets():
    fn_to_segment_path = lambda s: s + '_segment'
    assert main.gen_triplets([], fn_to_segment_path) == []
    assert main.gen_triplets(['foo'], fn_to_segment_path) == []
    #^ No negative samples present
    triplets = main.gen_triplets(['foo', 'bar', 'baz'], fn_to_segment_path)
    assert sorted(triplets, key=itemgetter('A', 'N')) == [
            {'A': 'bar', 'P': 'bar_segment', 'N': 'baz_segment'},
            {'A': 'bar', 'P': 'bar_segment', 'N': 'foo_segment'},
            {'A': 'baz', 'P': 'baz_segment', 'N': 'bar_segment'},
            {'A': 'baz', 'P': 'baz_segment', 'N': 'foo_segment'},
            {'A': 'foo', 'P': 'foo_segment', 'N': 'bar_segment'},
            {'A': 'foo', 'P': 'foo_segment', 'N': 'baz_segment'},
        ]

