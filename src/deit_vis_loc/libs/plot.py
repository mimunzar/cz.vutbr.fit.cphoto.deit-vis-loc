#!/usr/bin/env python3

import itertools as it
import functools as ft

import matplotlib.image as mpimg

import src.deit_vis_loc.libs.util as util


def im_on_axis(im, axis):
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(im['path']))
    return im


def im_grid(figure, grid, im_row_it):
    plot_im  = lambda r, c, im: im_on_axis(im, figure.add_subplot(grid[r, c]))
    plot_row = lambda r, im_it: tuple(it.starmap(ft.partial(plot_im, r), enumerate(im_it)))
    util.dorun(it.starmap(plot_row, enumerate(im_row_it)))

