#!/usr/bin/env python3

import pytest

import src.deit_vis_loc.libs.util as util


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
    assert list(util.prepend(1, []))           == [1]
    assert list(util.prepend(1, [2, 3]))       == [1, 2, 3]
    assert list(util.prepend(1, iter([2, 3]))) == [1, 2, 3]


def test_take():
    assert list(util.take(None, [1, 2, 3])) == [1, 2, 3]
    assert list(util.take(0,    [1, 2, 3])) == []
    assert list(util.take(2,    [1, 2, 3])) == [1, 2]
    assert list(util.take(42,   [1, 2, 3])) == [1, 2, 3]


def test_drop():
    assert list(util.drop(None, [1, 2, 3])) == [1, 2, 3]
    assert list(util.drop(0,    [1, 2, 3])) == [1, 2, 3]
    assert list(util.drop(2,    [1, 2, 3])) == [3]
    assert list(util.drop(42,   []))        == []


def test_first():
    with pytest.raises(StopIteration): util.first([])
    assert util.first([1, 2])       == 1
    assert util.first((1, 2))       == 1
    assert util.first(iter([1, 2])) == 1


def test_nth():
    with pytest.raises(StopIteration): util.nth(2, [])
    assert util.nth(0, [1, 2, 3])       == 1
    assert util.nth(1, (1, 2, 3))       == 2
    assert util.nth(2, iter([1, 2, 3])) == 3


def test_flatten():
    assert list(util.flatten([]))            == []
    assert list(util.flatten([[1, 2, 3]]))   == [1, 2, 3]
    assert list(util.flatten([[1], [2, 3]])) == [1, 2, 3]
    assert list(util.flatten([[[1, 2, 3]]])) == [[1, 2, 3]]


def test_rand_sample():
    assert list(util.rand_sample(1, [],        lambda: 0)) == []
    assert list(util.rand_sample(0, [1, 2, 3], lambda: 1)) == []
    assert list(util.rand_sample(1, [1, 2, 3], lambda: 1)) == []
    assert list(util.rand_sample(1, [1, 2, 3], lambda: 0)) == [1, 2, 3]


def test_pluck():
    assert util.pluck(['foo'], {'foo': 42, 'bar': 43}) == 42
    assert util.pluck(iter(['foo']), {'foo': 42, 'bar': 43}) == 42
    assert util.pluck(['foo', 'bar'], {'foo': 42, 'bar': 43}) == (42, 43)
    with pytest.raises(KeyError): util.pluck(['baz'], {'foo': 42, 'bar': 43})


def test_clamp():
    assert util.clamp(1, 3, 0) == 1
    assert util.clamp(1, 3, 1) == 1
    assert util.clamp(1, 3, 2) == 2
    assert util.clamp(1, 3, 3) == 3
    assert util.clamp(1, 3, 4) == 3


def test_complement():
    assert util.complement(lambda: True)()  == False
    assert util.complement(lambda: False)() == True
    assert util.complement(lambda i: 0 == i % 2)(1) == True
    assert util.complement(lambda i: 0 == i % 2)(2) == False


def test_compose():
    assert util.compose(next)(iter([1, 2, 3]))           == 1
    assert util.compose(next, iter)([1, 2, 3])           == 1
    assert util.compose(str, next, iter)([1, 2, 3])      == "1"
    assert util.compose(int, str, next, iter)([1, 2, 3]) == 1


def test_validator():
    assert util.make_validator('Fail', lambda: False)()    == (False, 'Fail')
    assert util.make_validator('Fail', lambda x: x > 0)(0) == (False, 'Fail')
    assert util.make_validator('Fail', lambda x: x > 0)(1) == (True, None)


def test_checker():
    checker = util.make_checker({
        'foo': util.make_validator('Failed foo', lambda x: x > 0),
        'bar': util.make_validator('Failed bar', lambda x: x > 0),
    })
    assert checker({'foo': 0, 'bar': 0, 'baz': 0}) == ('Failed foo', 'Failed bar',)
    assert checker({'foo': 0, 'bar': 1, 'baz': 0}) == ('Failed foo',)
    assert checker({'foo': 1, 'bar': 1, 'baz': 0}) == tuple()


def test_make_running_avg():
    ravg = util.make_running_avg()
    assert ravg(0) == 0
    assert ravg(2) == 1
    assert ravg(4) == 2
    assert ravg(6) == 3
    assert ravg(8) == 4

