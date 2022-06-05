#!/usr/bin/env python3

import itertools as it
import functools as ft
import os

import torch
import matplotlib.pyplot as plt

import src.deit_vis_loc.libs.util as util


LINE_WIDTH  = 2.5
TRAIN_COLOR = '#6BB4EF'
VAL_COLOR   = '#EFA66B'

def make_netsaver(output_dir, net):
    output_dir = os.path.expanduser(output_dir)
    net        = net.module if hasattr(net, 'module') else net
    def netsaver(epochstats):
        epoch = f'{epochstats["epoch"]:03}'
        torch.save(net, os.path.join(output_dir, f'net-{epoch}.torch'))
        return epochstats
    return netsaver


def iter_prependaxis(iterable, step=1):
    return zip(*zip(it.count(1, step), iterable))
    # => ((e1, e2, ...), (d1, d2, ...)


def plot_loss_on_axis(ax, tloss_it, vloss_it):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(*iter_prependaxis(tloss_it), color=TRAIN_COLOR, linewidth=LINE_WIDTH)
    ax.plot(*iter_prependaxis(vloss_it), color=VAL_COLOR,   linewidth=LINE_WIDTH)
    ax.legend(['train', 'val'])


def make_loss_plotter(outdir):
    outpath = os.path.join(os.path.expanduser(outdir), 'loss.svg')
    tlosses = []
    vlosses = []
    def loss_plotter(epochstats):
        tlosses.append(epochstats['train']['avg_loss'])
        vlosses.append(epochstats['val']['avg_loss'])
        fg, ax = plt.subplots()
        plot_loss_on_axis(ax, tlosses, vlosses)
        fg.savefig(outpath)
        plt.close(fg)
        return {'train': tlosses, 'val': vlosses}
    return loss_plotter


def plot_recall_on_axis(ax, trecall_it, vrecall_it, mine_nth_epoch):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall [%]')
    iter_r1   = ft.partial(map, ft.partial(util.pluck, ['at_1']))
    iter_r100 = ft.partial(map, ft.partial(util.pluck, ['at_100']))
    ax.plot(*iter_prependaxis(iter_r1(trecall_it), mine_nth_epoch),
            color=TRAIN_COLOR, linewidth=LINE_WIDTH)
    ax.plot(*iter_prependaxis(iter_r100(trecall_it), mine_nth_epoch),
            color=TRAIN_COLOR, linewidth=LINE_WIDTH, linestyle='dashed')
    ax.plot(*iter_prependaxis(iter_r1(vrecall_it), mine_nth_epoch),
            color=VAL_COLOR, linewidth=LINE_WIDTH)
    ax.plot(*iter_prependaxis(iter_r100(vrecall_it), mine_nth_epoch),
            color=VAL_COLOR, linewidth=LINE_WIDTH, linestyle='dashed')
    ax.legend(['train@1', 'train@100', 'val@1', 'val@100'])


def is_miningepoch(epoch, mine_nth_epoch):
    return 0 == (epoch - 1)%mine_nth_epoch


def make_recall_plotter(outdir, mine_nth_epoch):
    outpath    = os.path.join(os.path.expanduser(outdir), 'recall.svg')
    trecalls   = []
    vrecalls   = []
    iter_vperc = ft.partial(it.starmap, lambda x, y: (x, y*100))
    def recall_plotter(stats):
        if not is_miningepoch(stats['epoch'], mine_nth_epoch):
            return {'train': trecalls, 'val': vrecalls}

        trecalls.append(dict(iter_vperc(stats['train']['recall'].items())))
        vrecalls.append(dict(iter_vperc(stats['val']['recall']  .items())))

        fg, ax = plt.subplots()
        plot_recall_on_axis(ax, trecalls, vrecalls, mine_nth_epoch)
        fg.savefig(outpath)
        plt.close(fg)
        return {'train': trecalls, 'val': vrecalls}
    return recall_plotter

