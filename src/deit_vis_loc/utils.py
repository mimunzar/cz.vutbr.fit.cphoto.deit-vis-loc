#!/usr/bin/env python3

import itertools as it
import datetime  as dt


def partition(n, iterable):
    iterable = iter(iterable)
    return it.takewhile(lambda l: len(l) == n,
            (list(it.islice(iterable, n)) for _ in it.repeat(None)))


def to_segment_img(fpath):
    return fpath.replace('query_original_result', 'query_segments_result') \
        .replace('.jpg', '.png')


def make_validator(msg, fn_valid):
    return lambda *args: (True, None) if fn_valid(*args) else (False, msg)


def make_checker(dict_of_validators):
    def check_dict(dict_to_check):
        result = []
        for k, fn_val in dict_of_validators.items():
            success, msg = fn_val(dict_to_check[k])
            if not success:
                result.append(msg)
        return result
    return check_dict


def log(msg):
    d = dt.datetime.now(tz=dt.timezone.utc)
    print('[{}] {}'.format(d.strftime("%Y%m%dT%H%M%S"), msg))

