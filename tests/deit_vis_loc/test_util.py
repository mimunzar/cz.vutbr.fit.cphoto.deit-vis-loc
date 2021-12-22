#!/usr/bin/env python3

import src.deit_vis_loc.util as util

from math import pi


def test_partition():
    assert list(util.partition(1, [])) == []
    assert list(util.partition(4, [1, 2, 3])) == []
    assert list(util.partition(3, [1, 2, 3])) == [[1, 2, 3]]
    assert list(util.partition(2, [1, 2, 3])) == [[1, 2]]
    assert list(util.partition(1, [1, 2, 3])) == [[1], [2], [3]]


def test_partition_by():
    is_odd = lambda x: x % 2
    assert list(map(list, util.partition_by(is_odd, []))) == [[], []]
    assert list(map(list, util.partition_by(is_odd, [1]))) == [[1], []]
    assert list(map(list, util.partition_by(is_odd, [1, 2]))) == [[1], [2]]
    assert list(map(list, util.partition_by(is_odd, [1, 2, 3]))) == [[1, 3], [2]]


def test_prepend():
    assert list(util.prepend(1, [])) == [1]
    assert list(util.prepend(1, [2, 3])) == [1, 2, 3]
    assert list(util.prepend(1, iter([2, 3]))) == [1, 2, 3]


def test_subseq():
    assert list(util.subseq(42, [])) == []
    assert list(util.subseq(42, [42])) == [42]
    assert list(util.subseq(42, [41, 42])) == [42]
    assert list(util.subseq(42, [41, 42, 42])) == [42, 42]
    assert list(util.subseq(42, [41, 42, 42, 43])) == [42, 42]


def test_validator():
    assert util.make_validator('Fail', lambda: False)()    == (False, 'Fail')
    assert util.make_validator('Fail', lambda x: x > 0)(0) == (False, 'Fail')
    assert util.make_validator('Fail', lambda x: x > 0)(1) == (True, None)


def test_checker():
    checker = util.make_checker({
        'foo': util.make_validator('Failed foo', lambda x: x > 0),
        'bar': util.make_validator('Failed bar', lambda x: x > 0),
    })
    assert checker({'foo': 0, 'bar': 0, 'baz': 0}) == ['Failed foo', 'Failed bar']
    assert checker({'foo': 0, 'bar': 1, 'baz': 0}) == ['Failed foo']
    assert checker({'foo': 1, 'bar': 1, 'baz': 0}) == []


def test_circle_difference_rad():
    assert util.circle_difference_rad(2*pi, 2*pi) == 0
    assert util.circle_difference_rad(0,    2*pi) == 0
    assert util.circle_difference_rad(pi, 2*pi) == pi
    assert util.circle_difference_rad(2*pi, pi) == pi
    assert util.circle_difference_rad(0, 2*pi - pi/2) == pi/2
    assert util.circle_difference_rad(0,        pi/2) == pi/2

