#!/usr/bin/env python3

import matplotlib.pyplot as mpplt
import matplotlib.image as mpimg

from operator import itemgetter

import src.deit_vis_loc.utils as utils



def plot_image(axis, fpath, border=None):
    if border:
        for spine_pos in ['bottom','top', 'right', 'left']:
            axis.spines[spine_pos].set_color(border['color'])
            axis.spines[spine_pos].set_linewidth(border['width'])
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(fpath))


def plot_closest_distances(embedding_distance):
    query_path = embedding_distance['query_path']
    _, axis  = mpplt.subplots(1, 5, constrained_layout=True)
    plot_image(axis[0], query_path)

    list_of_segments = sorted(
            embedding_distance['segments'], key=itemgetter('distance'))
    for i in range(1, 5):
        segment_path = list_of_segments[i - 1]['path']
        border_color = 'green' if segment_path == utils.to_segment_path(query_path) else 'red'
        plot_image(axis[i], segment_path, border={'color': border_color, 'width': 3})

