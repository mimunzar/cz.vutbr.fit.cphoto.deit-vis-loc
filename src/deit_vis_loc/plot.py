#!/usr/bin/env python3

import matplotlib.image  as mpimg
import matplotlib.pyplot as mpplt

import collections as cl
import itertools   as it
import math        as ma
import operator    as op

import src.deit_vis_loc.util as util


def plot_img_on_axis(im, axis):
    fpath, color, _ = im
    for spine_pos in ['bottom','top', 'right', 'left']:
        axis.spines[spine_pos].set_color(color)
        axis.spines[spine_pos].set_linewidth(3)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(fpath))
    return (im, axis)


def ims_of_query_with_n_segments(n, name, segment_it):
    to_segment_img  = lambda s: (s['name'], 'green' if s['is_positive'] else 'red', s['distance'])
    to_segment_imgs = lambda l: map(to_segment_img, sorted(l, key=op.itemgetter('distance')))
    to_query_im     = lambda s: (s, 'black', 0)
    return util.prepend(to_query_im(name), it.islice(to_segment_imgs(segment_it), n))


def n_closest_segments(n, result_it):
    im_rows  = [ims_of_query_with_n_segments(n, r['query'], r['segments']) for r in result_it]
    figure   = mpplt.figure(constrained_layout=True)
    im_grid  = figure.add_gridspec(nrows=len(im_rows), ncols=n + 1, hspace=0.05, wspace=0.05)
    axis     = lambda r, c: figure.add_subplot(im_grid[r, c])
    plt_grid = (zip(ims, (axis(r, c) for c in it.count())) for r, ims in enumerate(im_rows))
    return [list(it.starmap(plot_img_on_axis, row)) for row in plt_grid]


def localization_percentile(segment_it):
    by_distancce = sorted(segment_it, key=op.itemgetter('distance'))
    positive_idx = next(i for i, s in enumerate(by_distancce, start=1) if s['is_positive'])
    return ma.ceil((positive_idx/len(by_distancce))*100)


def running_localization_percentage(percentile_it):
    grouped    = it.groupby(sorted(percentile_it))
    histogram  = cl.defaultdict(lambda: 0, {k: len(list(l)) for k, l in grouped})
    total_size = sum(histogram.values())
    percentage = lambda n: round((n/total_size)*100, ndigits=2)
    return [percentage(histogram[i]) for i in range(1, 101)]


def rank_of_correct_location(result_it):
    percentiles   = (localization_percentile(r['segments']) for r in result_it)
    bounded_add   = lambda x, y: min(x + y, 100)
    location_rank = list(it.accumulate(
        running_localization_percentage(percentiles), bounded_add))

    _, ax = mpplt.subplots()
    ax.set_xlabel('Rank of Correct Location (%)')
    ax.set_ylabel('% of Queries')
    ax.set_title('Localization Results')
    ax.grid(True, linestyle='--', linewidth=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 101)
    ax.set_xlim(0, 101)
    ax.plot(location_rank, linewidth=2.5)
    return location_rank

