#!/usr/bin/env python3

import functools as ft
import itertools as it
import datetime as dt

import src.deit_vis_loc.libs.util as util


LINE_WIDTH = 80

def msg(msg, prefix=''):
    d = dt.datetime.now(tz=dt.timezone.utc)
    return f'{prefix}[{d.strftime("%Y%m%dT%H%M%S")}] {msg}'


def fmt_table_col_width(t):
    n_cols  = len(t[0])
    sel_col = lambda i: map(ft.partial(util.nth, i), t)
    max_len = lambda c: max(map(len, c))
    return map(util.compose(max_len, sel_col), range(n_cols))


def fmt_table(t, line_width=LINE_WIDTH):
    col_width = tuple(fmt_table_col_width(t))
    fmt_cell  = lambda c, s, w: f'{s:<{w}} |' if 0 == c else f' {s:^{w}} |'
    fmt_row   = lambda r_it: it.starmap(fmt_cell, zip(it.count(), r_it, col_width))
    return map(util.compose(lambda r: ''.join(r).center(line_width), fmt_row), t)


def fmt_fraction(total, curr):
    max_len = len(str(total))
    return f'{curr:>{max_len}}/{total}'


def fmt_bar(bar_width, total, curr):
    curr = min(curr, total)
    fill = ('#'*round(curr/total*bar_width)).ljust(bar_width)
    return f'[{fill}]'


def make_progress_bar(bar_width, total, lwidth=LINE_WIDTH):
    bar = ft.partial(fmt_bar, bar_width, total)
    fra = ft.partial(fmt_fraction, total)
    def progress_bar(label, curr, speed, loss):
        prog = f'{label:>15}: {bar(curr)} {fra(curr):<15}'
        stat = f'{loss:.04f} loss, {speed:.02f} im/s'
        return f'{prog} ({stat})'.center(lwidth)
    return progress_bar

