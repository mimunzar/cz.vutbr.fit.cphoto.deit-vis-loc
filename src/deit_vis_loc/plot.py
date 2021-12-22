#!/usr/bin/env python3

import matplotlib.image  as mpimg
import matplotlib.pyplot as mpplt

import itertools as it
import math      as ma
import operator  as op

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


def ims_of_query_with_n_segments(n, name, segments_it):
    to_segment_img  = lambda s: (s['name'], 'green' if s['is_positive'] else 'red', s['distance'])
    to_segment_imgs = lambda l: map(to_segment_img, sorted(l, key=op.itemgetter('distance')))
    to_query_im     = lambda s: (s, 'black', 0)
    return util.prepend(to_query_im(name), it.islice(to_segment_imgs(segments_it), n))


def n_closest_segments(n, result_it):
    im_rows  = [ims_of_query_with_n_segments(n, r['query'], r['segments']) for r in result_it]
    figure   = mpplt.figure(constrained_layout=True)
    im_grid  = figure.add_gridspec(nrows=len(im_rows), ncols=n + 1, hspace=0.05, wspace=0.05)
    axis     = lambda r, c: figure.add_subplot(im_grid[r, c])
    plt_grid = (zip(ims, (axis(r, c) for c in it.count())) for r, ims in enumerate(im_rows))
    return [list(it.starmap(plot_img_on_axis, row)) for row in plt_grid]


def _index_of_anchor_segment(anchor, segments):
    return next(i for i, s in enumerate(segments) if s['is_positive'])


def percentage_of_localized_images(query_iterable, nsegments=6192):
    location_rank_perc  = lambda x: x*100/nsegments
    sorted_by_distance  = ((x['query'], sorted(x['segments'], key=op.itemgetter('distance'))) for x in query_iterable)
    anchor_segment_ids  = (_index_of_anchor_segment(*x) for x in sorted_by_distance)
    sorted_percentiles  = list(sorted(ma.ceil(location_rank_perc(x)) for x in anchor_segment_ids))
    percentile_buckets  = (list(util.subseq(x, sorted_percentiles)) for x in range(1, 101))
    percentile_lens     = it.accumulate(len(x) for x in percentile_buckets)
    percentile_percents = [location_rank_perc(x) for x in percentile_lens]

    _, ax = mpplt.subplots()
    ax.set_xlabel('Rank of Correct Location (%)')
    ax.set_ylabel('% of Queries')
    ax.set_title('Localization Results')
    ax.grid(True, linestyle='--', linewidth=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 101)
    ax.set_xlim(0, 101)
    ax.plot(percentile_percents, linewidth=2.5)

