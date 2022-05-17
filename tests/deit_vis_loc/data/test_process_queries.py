#!/usr/bin/env python3

import pytest

import src.deit_vis_loc.data.process_queries as process_queries


def test_parse_meta():
    with pytest.raises(Exception):
        process_queries.parse_meta('foo', [])
    with pytest.raises(Exception):
        process_queries.parse_meta('foo', ['MANUAL', 'foo'])
    assert process_queries.parse_meta('foo', [
        'MANUAL',
        '0.577994 -0.0274898 0.0177252',
        '46.1766',
        '7.85907',
        '2859.500000',
        '0.887981',
        '46.1766',
        '7.85907',
        '2859.500000',
        '0.887981']) == {
            'name'      : 'foo',
            'latitude'  : 46.1766,
            'longitude' : 7.85907,
            'elevation' : 2859.5,
            'yaw'       : 0.577994,
            'pitch'     : -0.0274898,
            'roll'      : 0.0177252,
            'FOV'       : 0.887981,
        }

