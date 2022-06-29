#!/usr/bin/env python3

import collections as cl
import functools as ft
import itertools as it
import operator as op
import random
from time import time


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


def sortby(f, iterable):
    return sorted(iterable, key=f)


def pluck(iterable, d):
    return op.itemgetter(*iterable)(d)


def update(k, f, d):
    d[k] = f(d[k]) if k in d else f()
    return d


def assoc(d, *kv):
    for k, v in kv:
        d[k] = v
    return d


def clamp(minimum, maximum, n):
    return max(minimum, min(maximum, n))


def identity(x):
    return x


def memoize(f):
    return ft.lru_cache(maxsize=None)(f)


def complement(f):
    return lambda *args: not f(*args)


def compose(*f_it):
    def compose2(f, g):
        return lambda *args: f(g(*args))
    return ft.reduce(compose2, f_it)


def juxt(*f_it):
    return lambda *args: tuple(map(lambda f: f(*args), f_it))


def repeatedly(f):
    return map(lambda _: f(), it.repeat(None))


def make_validator(msg, f):
    return lambda *args: (bool(f(*args)), msg)


def make_checker(validators):
    def check_dict(d):
        checks = it.starmap(lambda k, f: f(d[k]), validators.items())
        return tuple(map(second, filter(complement(first), checks)))
    return check_dict


def make_running_mean():
    i, avg = (0, 0)
    def running_avg(n):
        nonlocal i, avg
        i   = i + 1
        avg = (avg*(i - 1) + n)/i
        return avg
    return running_avg


def make_ims_sec(fn_fracsec=time):
    start = fn_fracsec()
    prev  = 0
    def ims_sec(now_ims, fn_fracsec=time):
        nonlocal start, prev
        elaps = max(1e-6, fn_fracsec() - start)
        start = start + elaps
        done  = now_ims - prev
        prev  = prev + done
        return done/elaps
    return ims_sec


def make_mean_ims_sec():
    ims_sec     = make_ims_sec()
    running_avg = make_running_mean()
    return lambda n: running_avg(ims_sec(n))

