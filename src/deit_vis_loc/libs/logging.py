#!/usr/bin/env python3

import functools as ft
import datetime as dt
import sys
import time

import src.deit_vis_loc.libs.util as util


def log(msg, start='', end='\n', file=sys.stdout):
    d = dt.datetime.now(tz=dt.timezone.utc)
    print(f'{start}[{d.strftime("%Y%m%dT%H%M%S")}] {msg}', end=end, file=file)


def format_fraction(n, d):
    d_len = len(str(d))
    return f'{n:>{d_len}}/{d}'


def format_bar(bar_width, total, curr):
    curr = min(curr, total)
    bar  = ('#'*round(curr/total*bar_width)).ljust(bar_width)
    return f'[{bar}] {format_fraction(curr, total)}'


def make_progress_bar(bar_width, total):
    bar = ft.partial(format_bar, bar_width, total)
    def progress_bar(stage, curr, speed, loss):
        return f'{stage:>15}: {bar(curr)}  ({loss:.2f} loss, {speed:.02f} im/s)'
    return progress_bar


def make_ims_sec(fn_epoch_secs=time.time):
    start = fn_epoch_secs()
    def ims_sec(done_ims, fn_epoch_secs=time.time):
        nonlocal start
        end    = fn_epoch_secs()
        result = done_ims/max(1e-6, end - start)
        start  = start +  max(1e-6, end - start)
        return result
    return ims_sec


def make_avg_ims_sec():
    ims_sec     = make_ims_sec()
    running_avg = util.make_running_avg()
    return lambda n: running_avg(ims_sec(n))

