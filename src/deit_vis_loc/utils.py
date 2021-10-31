#!/usr/bin/env python3

import itertools as it
import operator  as op


def partition(n, iterable):
    iterable = iter(iterable)
    return it.takewhile(lambda l: len(l) == n,
            (list(it.islice(iterable, n)) for _ in it.repeat(None)))


def to_segment_img(query_path):
    return query_path \
        .replace('query_original_result', 'query_segments_result') \
        .replace('.jpg', '.png')


def _plot_img(axis, fpath, border=None):
    import matplotlib.image as mpimg

    if border:
        for spine_pos in ['bottom','top', 'right', 'left']:
            axis.spines[spine_pos].set_color(border['color'])
            axis.spines[spine_pos].set_linewidth(border['width'])
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(fpath))


def plot_closest_segments(anchor_val):
    import matplotlib.pyplot as mpplt

    _, axis  = mpplt.subplots(1, 5, constrained_layout=True)
    _plot_img(axis[0], anchor_val['anchor'])

    list_of_segments = sorted(anchor_val['segments'], key=op.itemgetter(1))
    for i in range(1, 5):
        segment = list_of_segments[i - 1][0]
        border_color = 'green' if segment == to_segment_img(anchor_val['anchor']) else 'red'
        _plot_img(axis[i], segment, border={'color': border_color, 'width': 3})

