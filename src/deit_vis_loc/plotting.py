#!/usr/bin/env python3

import matplotlib.image  as mpimg
import matplotlib.pyplot as mpplt

import itertools as it
import math      as ma
import operator  as op

import src.deit_vis_loc.utils as utils


def _plot_img(axis, fpath, border=None):
    if border:
        for spine_pos in ['bottom','top', 'right', 'left']:
            axis.spines[spine_pos].set_color(border['color'])
            axis.spines[spine_pos].set_linewidth(border['width'])
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(fpath))


def _is_anchor_segment(anchor, segment):
    return segment == utils.to_segment_img(anchor)


def by_distance(segment_iterable):
    return sorted(segment_iterable, key=op.itemgetter('distance'))


def n_closest_segments(query_iterable, nsegments):
    query_iterable = list(query_iterable)
    fig  = mpplt.figure(constrained_layout=True)
    spec = fig.add_gridspec(nrows=len(query_iterable), ncols=nsegments + 1, hspace=0.05, wspace=0.05)

    for i, t in enumerate(query_iterable):
        _plot_img(fig.add_subplot(spec[i, 0]), t['anchor'])
        for j, s in zip(range(1, nsegments + 1), by_distance(t['segments'])):
            color = 'green' if _is_anchor_segment(t['anchor'], s['path']) else 'red'
            _plot_img(fig.add_subplot(spec[i, j]), s['path'], border={'color': color, 'width': 3})


def _index_of_anchor_segment(anchor, segments):
    return next(i for i, s in enumerate(segments)
            if _is_anchor_segment(anchor, s['path']))


def percentage_of_localized_images(query_iterable, nsegments=512):
    location_rank_perc  = lambda x: x*100/nsegments
    sorted_by_distance  = ((x['anchor'], by_distance(x['segments'])) for x in query_iterable)
    anchor_segment_ids  = (_index_of_anchor_segment(*x) for x in sorted_by_distance)
    sorted_percentiles  = list(sorted(ma.ceil(location_rank_perc(x)) for x in anchor_segment_ids))
    percentile_buckets  = (list(utils.subseq(x, sorted_percentiles)) for x in range(1, 101))
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

