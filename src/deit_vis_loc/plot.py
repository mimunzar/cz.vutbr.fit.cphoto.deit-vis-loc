#!/usr/bin/env python3

import matplotlib.image  as mpimg
import matplotlib.pyplot as mpplt

import collections as cl
import itertools   as it
import functools   as ft
import math        as ma

import src.deit_vis_loc.util as util


def plot_im_on_axis(im, axis):
    fpath, color, dist = im
    for spine_pos in ['bottom','top', 'right', 'left']:
        axis.spines[spine_pos].set_color(color)
        axis.spines[spine_pos].set_linewidth(3)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    if 0 != dist: axis.text(7, 28, round(dist, 4), bbox={'facecolor': 'white'})
    axis.imshow(mpimg.imread(fpath))
    return (im, axis)


def im_and_n_closest_segments(n, name, segment_it):
    to_query_im    = lambda s: (s, 'black', 0)
    to_segment_img = lambda s: (s['segment'], 'green' if s['is_pos'] else 'red', s['dist'])
    return util.prepend(to_query_im(name), util.take(n, map(to_segment_img, segment_it)))


def plot_n_closest_segments(n, results_it):
    rows_it  = tuple(it.starmap(ft.partial(im_and_n_closest_segments, n), results_it))
    figure   = mpplt.figure(constrained_layout=True)
    im_grid  = figure.add_gridspec(nrows=len(rows_it), ncols=n + 1, hspace=0.05, wspace=0.05)
    plot_im  = lambda r, c, im: plot_im_on_axis(im, figure.add_subplot(im_grid[r, c]))
    plot_row = lambda r, im_it: tuple(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    return tuple(it.starmap(plot_row, enumerate(rows_it)))


def localization_percentile(segment_it):
    segment_it = tuple(segment_it)
    pos_idx    = next(i for i, s in enumerate(segment_it, start=1) if s['is_pos'])
    return ma.ceil((pos_idx/len(segment_it))*100)


def histogram(iterable):
    grouped = it.groupby(sorted(iterable))
    return cl.defaultdict(lambda: 0, it.starmap(lambda k, v: (k, len(tuple(v))), grouped))


def iter_histogram_perc(hist, rng):
    total_size = sum(hist.values())
    if 0 == total_size: return iter(())
    return map(lambda i: round((hist[i]/total_size)*100, ndigits=2), rng)


def plot_rank_of_correct_location(results_it):
    percentiles   = map(localization_percentile, map(util.second, results_it))
    bounded_add   = lambda x, y: min(x + y, 100)
    location_rank = tuple(it.accumulate(
        iter_histogram_perc(histogram(percentiles), range(1, 101)), bounded_add))

    _, axis = mpplt.subplots()
    axis.set_xlabel('Rank of Correct Location (%)')
    axis.set_ylabel('% of Queries')
    axis.set_title('Localization Results')
    axis.grid(True, linestyle='--', linewidth=2)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_ylim(0, 101)
    axis.set_xlim(1, 101)
    axis.plot(location_rank, linewidth=2.5)
    return location_rank

