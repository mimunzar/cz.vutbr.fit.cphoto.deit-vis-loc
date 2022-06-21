#!/usr/bin/env python3

import math
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


def scale(ratio, im):
    dim_x = round(im.size[0]*ratio)
    dim_y = round(im.size[1]*ratio)
    return im.resize((dim_x, dim_y))


def scale_to_fit(input_size, im):
    return scale(input_size/max(im.size), im)


RENDER_FOV = math.radians(60)

def scale_by_fov(input_size, image_fov, im):
    im_re_fov = image_fov/RENDER_FOV
    re_im_px  = input_size/max(im.size)
    return scale(re_im_px*im_re_fov, im)
    #^ Scale image first to be the same size as renders and  than  scale  it  by
    # its FOV.   Avoid  upsampling  by  computing  the  scale  factor  at  once.


def center_crop(input_size, im):
    top_x = round((im.size[0] - input_size)/2)
    top_y = round((im.size[1] - input_size)/2)
    return im.crop((top_x, top_y, top_x + input_size, top_y + input_size))


def pad_to_square(input_size, im):
    new_im  = Image.new('RGB', (input_size, input_size))
    x_coord = round((input_size - im.size[0])/2)
    y_coord = round((input_size - im.size[1])/2)
    new_im.paste(im.convert('RGB'), (x_coord, y_coord))
    return new_im

