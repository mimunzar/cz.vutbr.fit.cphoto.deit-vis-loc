#!/usr/bin/env python3

import pytest

import src.deit_vis_loc.data.process_renders_sparse as process_renders_sparse


def test_parse_line():
    with pytest.raises(Exception):
        process_renders_sparse.parse_line('/foo/bar', [])
    with pytest.raises(Exception):
        process_renders_sparse.parse_line('/foo/bar',['foo', 'bar', 'baz'])
    assert process_renders_sparse.parse_line('/foo/bar', [
        'segment',
        'query',
        '46.2173',
        '10.1663',
        '459.5',
        '0',
        '1.5708',
        '1.5708',
        '1.0472']) == {
                'path'      : '/foo/bar/segment.jpg',
                'name'      : 'segment',
                'query'     : 'query',
                'latitude'  : 46.2173,
                'longitude' : 10.1663,
                'elevation' : 459.5,
                'yaw'       : 0,
                'pitch'     : 0,
                'roll'      : 0,
            }

