#!/usr/bin/env python3

import collections as cl
import functools as ft
import itertools as it
import operator as op
import random
import time


def partition(n, iterable, strict=True):
    iterable = iter(iterable)
    if strict:
        return zip(*[iterable]*n)
    return iter(lambda: tuple(take(n, iterable)), ())


def partition_by(pred, iterable):
    i1, i2 = it.tee(iterable)
    return (filter(pred, i1), it.filterfalse(pred, i2))


def prepend(x, iterable):
    return it.chain([x], iterable)


def take(n, iterable):
    return it.islice(iterable, n)


def take_last(n, iterable):
    return cl.deque(iterable, maxlen=n)


def drop(n, iterable):
    return it.islice(iterable, n, None)


def dorun(iterable):
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


def update(k, f, d):
    d[k] = f(d[k]) if k in d else f()
    return d


def clamp(minimum, maximum, n):
    return max(minimum, min(maximum, n))


def identity(x):
    return x


def memoize(f):
    return ft.lru_cache(maxsize=None)(f)


def memoize_tensor(device, f):
    mem = {}
    def cached_fn(*args):
        if args in mem:
            return mem[args].to(device)
        result    = f(*args)
        mem[args] = result.cpu()
        return result
    return cached_fn


def complement(f):
    return lambda *args: not f(*args)


def _compose2(f, g):
    return lambda *args: f(g(*args))


def compose(*f):
    return ft.reduce(_compose2, f)


def make_validator(msg, f):
    return lambda *args: (bool(f(*args)), msg)


def make_checker(validators):
    def check_dict(d):
        checks = it.starmap(lambda k, f: f(d[k]), validators.items())
        return tuple(map(second, filter(complement(first), checks)))
    return check_dict


def epoch_secs():
    return int(time.time())


def make_running_avg():
    idx, ravg = (0, 0)
    def running_avg(n):
        nonlocal idx, ravg
        idx  = idx + 1
        ravg = ravg*(idx - 1)/idx + n/idx
        return ravg
    return running_avg

