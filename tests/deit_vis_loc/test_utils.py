#!/usr/bin/env python3

import src.deit_vis_loc.utils as utils


def test_partition():
    assert list(utils.partition(1, [])) == []
    assert list(utils.partition(4, [1, 2, 3])) == []
    assert list(utils.partition(3, [1, 2, 3])) == [[1, 2, 3]]
    assert list(utils.partition(2, [1, 2, 3])) == [[1, 2]]
    assert list(utils.partition(1, [1, 2, 3])) == [[1], [2], [3]]


def test_validator():
    assert utils.make_validator('Fail', lambda: False)()    == (False, 'Fail')
    assert utils.make_validator('Fail', lambda x: x > 0)(0) == (False, 'Fail')
    assert utils.make_validator('Fail', lambda x: x > 0)(1) == (True, None)


def test_checker():
    checker = utils.make_checker({
        'foo': utils.make_validator('Failed foo', lambda x: x > 0),
        'bar': utils.make_validator('Failed bar', lambda x: x > 0),
    })
    assert checker({ 'foo': 0, 'bar': 0, 'baz': 0 }) == ['Failed foo', 'Failed bar']
    assert checker({ 'foo': 0, 'bar': 1, 'baz': 0 }) == ['Failed foo']
    assert checker({ 'foo': 1, 'bar': 1, 'baz': 0 }) == []

