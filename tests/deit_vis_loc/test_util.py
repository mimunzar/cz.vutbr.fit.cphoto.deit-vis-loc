#!/usr/bin/env python3

from math import pi

import pytest

import src.deit_vis_loc.util as util



def test_partition():
    assert list(util.partition(1, [])) == []
    assert list(util.partition(4, [1, 2, 3])) == []
    assert list(util.partition(3, [1, 2, 3])) == [(1, 2, 3)]
    assert list(util.partition(2, [1, 2, 3])) == [(1, 2)]
    assert list(util.partition(1, [1, 2, 3])) == [(1,), (2,), (3,)]


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


def test_first():
    with pytest.raises(StopIteration): util.first([])
    assert util.first([1, 2]) == 1
    assert util.first((1, 2)) == 1
    assert util.first(iter([1, 2])) == 1


def test_take():
    assert list(util.take(3, [])) == []
    assert list(util.take(0, [1, 2, 3])) == []
    assert list(util.take(2, [1, 2, 3])) == [1, 2]
    assert list(util.take(None, [1, 2, 3])) == [1, 2, 3]


def test_flatten():
    assert list(util.flatten([])) == []
    assert list(util.flatten([[1, 2, 3]])) == [1, 2, 3]
    assert list(util.flatten([[1], [2, 3]])) == [1, 2, 3]
    assert list(util.flatten([[[1, 2, 3]]])) == [[1, 2, 3]]


def test_pluck():
    assert util.pluck(['foo'], {'foo': 42, 'bar': 43}) == 42
    assert util.pluck(iter(['foo']), {'foo': 42, 'bar': 43}) == 42
    assert util.pluck(['foo', 'bar'], {'foo': 42, 'bar': 43}) == (42, 43)
    with pytest.raises(KeyError): util.pluck(['baz'], {'foo': 42, 'bar': 43})


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


def test_print_progress():
    assert util.progress_bar(1, 1, 0) == '[ ] 0/1'
    assert util.progress_bar(1, 1, 1) == '[#] 1/1'
    assert util.progress_bar(1, 1, 2) == '[#] 1/1'

    assert util.progress_bar(5, 1, 0)    == '[     ] 0/1'
    assert util.progress_bar(5, 1, 0.33) == '[##   ] 0.33/1'
    assert util.progress_bar(5, 1, 1)    == '[#####] 1/1'

    assert util.progress_bar(10, 5, 0) == '[          ] 0/5'
    assert util.progress_bar(10, 5, 1) == '[##        ] 1/5'
    assert util.progress_bar(10, 5, 5) == '[##########] 5/5'


def test_format_progress():
    f = util.make_progress_formatter(bar_width=1, total=1)
    assert f(stage='Foo',    curr=0, speed=0)      == '            Foo: [ ] 0/1  (0.00 im/s)'
    assert f(stage='Foo',    curr=1, speed=0.5)    == '            Foo: [#] 1/1  (0.50 im/s)'
    assert f(stage='FooFoo', curr=1, speed=0.5)    == '         FooFoo: [#] 1/1  (0.50 im/s)'
    assert f(stage='Foo',    curr=1, speed=1000.5) == '            Foo: [#] 1/1  (1000.50 im/s)'


def test_format_fraction():
    assert util.format_fraction(1, 1)   == '1/1'
    assert util.format_fraction(1, 10)  == ' 1/10'
    assert util.format_fraction(1, 100) == '  1/100'


def test_circle_difference_rad():
    assert util.circle_difference_rad(2*pi, 2*pi) == 0
    assert util.circle_difference_rad(0,    2*pi) == 0
    assert util.circle_difference_rad(pi, 2*pi) == pi
    assert util.circle_difference_rad(2*pi, pi) == pi
    assert util.circle_difference_rad(0, 2*pi - pi/2) == pi/2
    assert util.circle_difference_rad(0,        pi/2) == pi/2


def test_make_running_avg():
    ravg = util.make_running_avg()
    assert ravg(0) == 0
    assert ravg(2) == 1
    assert ravg(4) == 2
    assert ravg(6) == 3
    assert ravg(8) == 4


def test_make_ims_sec():
    ims_sec = util.make_ims_sec(lambda: 0)
    assert ims_sec(1, lambda: 1) == 1   # 1 seconds diff
    assert ims_sec(5, lambda: 6) == 1   # 5 seconds diff
    assert ims_sec(5, lambda: 6) == 5e6 # 0 seconds diff


def test_total_triplets():
    assert util.im_triplets(1, 1, 1) == 1
    assert util.im_triplets(1, 1, 2) == 2
    assert util.im_triplets(2, 1, 1) == 3
    assert util.im_triplets(1, 2, 2) == 4
    assert util.im_triplets(2, 2, 2) == 12

