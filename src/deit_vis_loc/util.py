#!/usr/bin/env python3

import datetime  as dt
import functools as ft
import itertools as it
import math      as ma
import operator  as op
import time


def partition(n, iterable):
    iterable = iter(iterable)
    return it.takewhile(lambda l: len(l) == n,
            (tuple(it.islice(iterable, n)) for _ in it.repeat(None)))


def partition_by(pred, iterable):
    i1, i2 = it.tee(iterable)
    return (filter(pred, i1), it.filterfalse(pred, i2))


def prepend(x, iterable):
    return it.chain([x], iterable)


def take(n, iterable):
    return it.islice(iterable, n)


def flatten(iterable):
    return it.chain.from_iterable(iterable)


def pluck(iterable, d):
    return op.itemgetter(*iterable)(d)


def memoize(f):
    return ft.lru_cache(maxsize=None)(f)


def make_validator(msg, fn_valid):
    return lambda *args: (True, None) if fn_valid(*args) else (False, msg)


def make_checker(validators):
    def check_dict(to_check):
        vals = [fn_val(to_check[k]) for k, fn_val in validators.items()]
        return [msg for succ, msg in vals if not succ]
    return check_dict


def log(msg):
    d = dt.datetime.now(tz=dt.timezone.utc)
    print('[{time}] {message}'.format(time=d.strftime("%Y%m%dT%H%M%S"), message=msg))


def progress_bar(bar_width, total, curr):
    curr = min(curr, total)
    bar  = ('#'*round(curr/total*bar_width)).ljust(bar_width)
    return f'[{bar}] {curr}/{total}'


def epoch_secs():
    return int(time.time())


def make_ims_sec(fn_epoch_secs=epoch_secs):
    start = fn_epoch_secs()
    def ims_sec(done_ims, fn_epoch_secs=epoch_secs):
        nonlocal start
        end    = fn_epoch_secs()
        result = done_ims/max(1, end - start)
        start  = end
        return result
    return ims_sec


def circle_difference_rad(l_rad, r_rad):
    distance = abs(l_rad - r_rad) % (2*ma.pi)
    return 2*ma.pi - distance if distance > ma.pi else distance
    #^ If distance is longer than half circle, there is a shorter way


def make_running_avg():
    idx  = 0
    ravg = 0
    def running_avg(n):
        nonlocal idx, ravg
        idx  = idx + 1
        ravg = ravg*(idx - 1)/idx + n/idx
        return ravg
    return running_avg

