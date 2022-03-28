#!/usr/bin/env python3

import pytest

import src.deit_vis_loc.preprocessing.dataset_im as dataset_im


def test_parse_info_file():
    with pytest.raises(Exception):
        dataset_im.parse_info_file('foo', [])
    with pytest.raises(Exception):
        dataset_im.parse_info_file('foo', ['MANUAL', 'foo'])
    assert dataset_im.parse_info_file('foo', [
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

