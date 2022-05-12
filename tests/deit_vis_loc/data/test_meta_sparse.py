#!/usr/bin/env python3

import pytest

import src.deit_vis_loc.data.meta_sparse as meta_sparse


def test_parse_line():
    with pytest.raises(Exception):
        meta_sparse.parse_line([])
    with pytest.raises(Exception):
        meta_sparse.parse_line(['foo', 'bar', 'baz'])
    assert meta_sparse.parse_line([
        'segment',
        'query',
        '46.2173',
        '10.1663',
        '459.5',
        '0',
        '1.5708',
        '1.5708',
        '1.0472']) == {
                'name'      : 'segment',
                'query'     : 'query',
                'latitude'  : 46.2173,
                'longitude' : 10.1663,
                'elevation' : 459.5,
                'yaw'       : 0,
                'pitch'     : 0,
                'roll'      : 0,
            }

