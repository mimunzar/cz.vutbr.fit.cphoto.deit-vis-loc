#!/usr/bin/env python3

import datetime  as dt
import functools as ft
import itertools as it
import math      as ma
import operator  as op
import time
import sys


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
    return it.islice(iterable, n if n is not None else 0)


def drop(n, iterable):
    return it.islice(iterable, n, None)


def first(iterable):
    return next(iter(iterable))


def nth(n, iterable):
    return first(drop(n, iterable))


def second(iterable):
    return nth(1, iterable)


def flatten(iterable):
    return it.chain.from_iterable(iterable)


def pluck(iterable, d):
    return op.itemgetter(*iterable)(d)


def memoize(f):
    return ft.lru_cache(maxsize=None)(f)


def make_validator(msg, fn_valid):
    return lambda *args: (True, None) if fn_valid(*args) else (False, msg)


def make_checker(validators):
    def check_dict(d):
        checks = it.starmap(lambda k, f: f(d[k]), validators.items())
        return tuple(map(second, it.filterfalse(first, checks)))
    return check_dict


def log(msg, start='', end='\n', file=sys.stdout):
    d = dt.datetime.now(tz=dt.timezone.utc)
    print(f'{start}[{d.strftime("%Y%m%dT%H%M%S")}] {msg}', end=end, file=file)


def format_fraction(n, d):
    d_len = len(str(d))
    return f'{n:>{d_len}}/{d}'


def progress_bar(bar_width, total, curr):
    curr = min(curr, total)
    bar  = ('#'*round(curr/total*bar_width)).ljust(bar_width)
    return f'[{bar}] {format_fraction(curr, total)}'


def make_progress_formatter(bar_width, total):
    prog_bar = ft.partial(progress_bar, bar_width, total)
    def progress_formatter(stage, curr, speed):
        return f'{stage:>15}: {prog_bar(curr)}  ({speed:.02f} im/s)'
    return progress_formatter


def epoch_secs():
    return int(time.time())


def make_ims_sec(fn_epoch_secs=time.time):
    start = fn_epoch_secs()
    def ims_sec(done_ims, fn_epoch_secs=time.time):
        nonlocal start
        end    = fn_epoch_secs()
        result = done_ims/max(1e-6, end - start)
        start  = start +  max(1e-6, end - start)
        return result
    return ims_sec


def make_running_avg():
    idx, ravg = (0, 0)
    def running_avg(n):
        nonlocal idx, ravg
        idx  = idx + 1
        ravg = ravg*(idx - 1)/idx + n/idx
        return ravg
    return running_avg


def make_avg_ims_sec():
    ims_sec     = make_ims_sec()
    running_avg = make_running_avg()
    return lambda n: running_avg(ims_sec(n))


def circle_difference_rad(l_rad, r_rad):
    distance = abs(l_rad - r_rad) % (2*ma.pi)
    return 2*ma.pi - distance if distance > ma.pi else distance
    #^ If distance is longer than half circle, there is a shorter way


def im_triplets(n_ims, n_pos, n_neg):
    card_pos = n_pos
    card_neg = (n_ims - 1)*(n_pos + n_neg) + n_neg
    return card_neg*card_pos
    #^ Cardinality of product of pos and neg sets over input images

