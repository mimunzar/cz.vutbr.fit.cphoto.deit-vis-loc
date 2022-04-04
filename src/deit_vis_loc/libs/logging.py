#!/usr/bin/env python3

import functools as ft
import datetime as dt
import sys
import time


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
    def progress_formatter(stage, curr, speed, loss):
        return f'{stage:>15}: {prog_bar(curr)}  ({loss:.2f} loss, {speed:.02f} im/s)'
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

