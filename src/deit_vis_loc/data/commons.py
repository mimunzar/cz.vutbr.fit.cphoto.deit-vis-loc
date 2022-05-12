#!/usr/bin/env python3

import re


def parse_into(ordt, csv_it):
    return {k: f(v) for (k, f), v in zip(ordt.items(), csv_it)}


WHITE_CHARS = re.compile(r'\s+')

def del_white(s):
    return WHITE_CHARS.sub('', s)


def iter_csv_lines(s):
    return del_white(s).replace(';', ',').split(',')


def parse_csv_file(fn_parse_line, fpath):
    try:
        return map(fn_parse_line, map(iter_csv_lines, open(fpath)))
    except Exception as ex:
        raise ValueError(f'Failed to parse {fpath} ({ex})')

