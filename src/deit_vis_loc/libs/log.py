#!/usr/bin/env python3

import functools as ft
import itertools as it
import datetime as dt
import sys
import time

import src.deit_vis_loc.libs.util as util


LINE_WIDTH = 80

def log(msg, start='', end='\n', file=sys.stdout):
    d = dt.datetime.now(tz=dt.timezone.utc)
    print(f'{start}[{d.strftime("%Y%m%dT%H%M%S")}] {msg}', end=end, file=file)


def fmt_table_col_width(t):
    n_cols  = len(t[0])
    sel_col = lambda i: map(ft.partial(util.nth, i), t)
    max_len = lambda c: max(map(len, c))
    return map(util.compose(max_len, sel_col), range(n_cols))


def fmt_table(t, lwidth=LINE_WIDTH):
    col_width = tuple(fmt_table_col_width(t))
    fmt_cell  = lambda c, s, w: f'{s:<{w}} |' if 0 == c else f' {s:^{w}} |'
    fmt_row   = lambda r_it: it.starmap(fmt_cell, zip(it.count(), r_it, col_width))
    return map(util.compose(lambda r: ''.join(r).center(lwidth), fmt_row), t)


def fmt_fraction(n, d):
    d_len = len(str(d))
    return f'{n:>{d_len}}/{d}'


def fmt_bar(bar_width, total, curr):
    curr = min(curr, total)
    bar  = ('#'*round(curr/total*bar_width)).ljust(bar_width)
    return f'[{bar}] {fmt_fraction(curr, total)}'


def make_progress_bar(bar_width, total, lwidth=LINE_WIDTH):
    bar = ft.partial(fmt_bar, bar_width, total)
    def progress_bar(stage, curr, speed, loss):
        return f'{stage}: {bar(curr)}  ({loss:.2f} loss, {speed:.02f} im/s)'.center(lwidth)
    return progress_bar


def make_ims_sec(fn_epoch_secs=time.time):
    start = fn_epoch_secs()
    def ims_sec(n_im, fn_epoch_secs=time.time):
        nonlocal start
        end    = fn_epoch_secs()
        elaps  = max(1e-6, end - start)
        start  = start + elaps
        return n_im/elaps
    return ims_sec


def make_avg_ims_sec():
    ims_sec     = make_ims_sec()
    running_avg = util.make_running_avg()
    return lambda n: running_avg(ims_sec(n))

