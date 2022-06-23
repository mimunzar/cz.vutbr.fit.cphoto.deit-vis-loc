#!/usr/bin/env python3

import re


def values_into(ordt, csv_it):
    return {k: f(v) for (k, f), v in zip(ordt.items(), csv_it)}


def iter_file(fn_parse_line, fpath):
    whites_re = re.compile(r'\s+')
    iter_vals = lambda s: whites_re.sub('', s).replace(';', ',').split(',')
    try:
        return map(fn_parse_line, map(iter_vals, open(fpath)))
    except Exception as ex:
        raise ValueError(f'Failed to parse {fpath} ({ex})')

