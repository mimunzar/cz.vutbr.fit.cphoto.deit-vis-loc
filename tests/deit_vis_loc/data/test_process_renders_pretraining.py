#!/usr/bin/env python3

import pytest

import src.deit_vis_loc.data.process_renders_pretraining as process_renders_pretraining


def test_parse_line():
    with pytest.raises(Exception):
        process_renders_pretraining.parse_line([])
    with pytest.raises(Exception):
        process_renders_pretraining.parse_line(['foo', 'bar', 'baz'])
    assert process_renders_pretraining.parse_line([
        '0000001',
        '28488116812_f5a57ca0f6_k',
        '46.2173',
        '10.1663',
        '439.500000',
        '-0.272811',
        '0.243122',
        '0.0148168',
        '0.549165']) == {
            'name'      : '28488116812_f5a57ca0f6_k',
            'query'     : '28488116812_f5a57ca0f6_k',
            'latitude'  : 46.2173,
            'longitude' : 10.1663,
            'elevation' : 439.5,
            'yaw'       : -0.272811,
            'pitch'     : 0.243122,
            'roll'      : 0.0148168,
        }

