#!/usr/bin/env python3

import matplotlib.image  as mpimg
import matplotlib.pyplot as mpplt

import operator as op

import src.deit_vis_loc.utils as utils


def _plot_img(axis, fpath, border=None):
    if border:
        for spine_pos in ['bottom','top', 'right', 'left']:
            axis.spines[spine_pos].set_color(border['color'])
            axis.spines[spine_pos].set_linewidth(border['width'])
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(fpath))


def is_correct_segment(anchor, segment):
    return segment == utils.to_segment_img(anchor)


def n_closest_segments(gen_tested, nsegments):
    list_of_tested = list(gen_tested)
    fig  = mpplt.figure(constrained_layout=True)
    spec = fig.add_gridspec(len(list_of_tested), nsegments + 1, hspace=0.05, wspace=0.05)

    for i, test in enumerate(list_of_tested):
        _plot_img(fig.add_subplot(spec[i, 0]), test['anchor'])
        segments_by_distance = sorted(test['segments'], key=op.itemgetter('distance'))
        for j, segment in zip(range(nsegments), segments_by_distance):
            axis  = fig.add_subplot(spec[i, j + 1])
            color = 'green' if is_correct_segment(test['anchor'], segment) else 'red'
            _plot_img(axis, segment['path'], border={'color': color, 'width': 3})

