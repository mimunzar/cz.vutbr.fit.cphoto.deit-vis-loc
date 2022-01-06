#!/usr/bin/env python3

import itertools as it
import datetime  as dt
import math      as ma


def partition(n, iterable):
    iterable = iter(iterable)
    return it.takewhile(lambda l: len(l) == n,
            (list(it.islice(iterable, n)) for _ in it.repeat(None)))


def partition_by(pred, iterable):
    i1, i2 = it.tee(iterable)
    return (filter(pred, i1), it.filterfalse(pred, i2))


def prepend(x, iterable):
    return it.chain([x], iterable)


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


def circle_difference_rad(l_rad, r_rad):
    distance = abs(l_rad - r_rad) % (2*ma.pi)
    return 2*ma.pi - distance if distance > ma.pi else distance
    #^ If distance is longer than half circle, there is a shorter way

