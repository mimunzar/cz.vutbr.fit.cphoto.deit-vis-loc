#!/usr/bin/env python3

import collections as cl
import functools as ft
import itertools as it
import operator as op
import random


def partition(n, iterable):
    iterable = iter(iterable)
    return it.takewhile(lambda l: len(l) == n,
            (tuple(take(n, iterable)) for _ in it.repeat(None)))


def partition_by(pred, iterable):
    i1, i2 = it.tee(iterable)
    return (filter(pred, i1), it.filterfalse(pred, i2))


def prepend(x, iterable):
    return it.chain([x], iterable)


def take(n, iterable):
    return it.islice(iterable, n)


def drop(n, iterable):
    return it.islice(iterable, n, None)


def consume(iterable):
    cl.deque(iterable, maxlen=0)


def first(iterable):
    return next(iter(iterable))


def nth(n, iterable):
    return first(drop(n, iterable))


def second(iterable):
    return nth(1, iterable)


def flatten(iterable):
    return it.chain.from_iterable(iterable)


def rand_sample(prob, iterable, fn_rand=random.random):
    return filter(lambda _: fn_rand() < prob, iterable)


def pluck(iterable, d):
    return op.itemgetter(*iterable)(d)


def memoize(f):
    return ft.lru_cache(maxsize=None)(f)


def clamp(minimum, maximum, n):
    return max(minimum, min(maximum, n))


def complement(f):
    return lambda *args: not f(*args)


def _compose2(f, g):
    return lambda *args: f(g(*args))


def compose(*f):
    return ft.reduce(_compose2, f)


def make_validator(msg, fn_valid):
    return lambda *args: (True, None) if fn_valid(*args) else (False, msg)


def make_checker(validators):
    def check_dict(d):
        checks = it.starmap(lambda k, f: f(d[k]), validators.items())
        return tuple(map(second, filter(complement(first), checks)))
    return check_dict

