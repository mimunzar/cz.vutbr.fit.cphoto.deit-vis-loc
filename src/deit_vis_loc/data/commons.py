#!/usr/bin/env python3

import re
from PIL import Image


def parse_into(ordt, csv_it):
    return {k: f(v) for (k, f), v in zip(ordt.items(), csv_it)}


WHITE_CHARS = re.compile(r'\s+')

def del_white(s):
    return WHITE_CHARS.sub('', s)


def iter_csv_lines(s):
    return del_white(s).replace(';', ',').split(',')


def iter_csv_file(fn_parse_line, fpath):
    try:
        return map(fn_parse_line, map(iter_csv_lines, open(fpath)))
    except Exception as ex:
        raise ValueError(f'Failed to parse {fpath} ({ex})')


def pad_to_square(res, im):
    n_im = Image.new('RGB', (res, res))
    n_im.paste(im.convert('RGB'), [(res - x)//2 for x in im.size])
    return n_im


def resize_keep_ratio(res, im):
    ratio = res/max(im.size)
    return im.resize([int(ratio*x) for x in im.size], Image.BICUBIC)

