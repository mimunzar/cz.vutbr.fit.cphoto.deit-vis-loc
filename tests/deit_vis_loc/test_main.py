#!/usr/bin/env python3

import src.deit_vis_loc.main as main

from operator import itemgetter


def test_generate_triplets():
    fn_to_segment_path = lambda s: s + '_segment'
    assert list(main.gen_triplets([], fn_to_segment_path)) == []
    assert list(main.gen_triplets(['foo'], fn_to_segment_path)) == []
    #^ No negative samples present
    triplets = main.gen_triplets(['foo', 'bar', 'baz'], fn_to_segment_path)
    assert sorted(triplets, key=itemgetter('anchor', 'negative')) == [
        {'anchor': 'bar', 'positive': 'bar_segment', 'negative': 'baz_segment'},
        {'anchor': 'bar', 'positive': 'bar_segment', 'negative': 'foo_segment'},
        {'anchor': 'baz', 'positive': 'baz_segment', 'negative': 'bar_segment'},
        {'anchor': 'baz', 'positive': 'baz_segment', 'negative': 'foo_segment'},
        {'anchor': 'foo', 'positive': 'foo_segment', 'negative': 'bar_segment'},
        {'anchor': 'foo', 'positive': 'foo_segment', 'negative': 'baz_segment'},
    ]

