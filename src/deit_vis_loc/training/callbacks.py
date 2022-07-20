#!/usr/bin/env python3

import itertools as it
import functools as ft
import os
import pickle

import torch
import matplotlib.pyplot as plt

import src.deit_vis_loc.libs.util as util


def make_saver_net(outdir, net):
    outdir = os.path.expanduser(outdir)
    net    = net.module if hasattr(net, 'module') else net
    def netsaver(stats):
        epoch = f'{stats["epoch"]:03}'
        torch.save(net, os.path.join(outdir, f'net-{epoch}.torch'))
        return stats
    return netsaver


def make_saver_stats(outdir):
    outpath  = os.path.join(os.path.expanduser(outdir), 'stats.bin')
    stats_it = []
    def saver_stats(stats):
        stats_it.append(stats)
        with open(outpath, 'wb') as f:
            pickle.dump(stats_it, f)
        return stats
    return saver_stats


def iter_prependaxis(iterable, step=1):
    return zip(*zip(it.count(1, step), iterable))
    # => ((e1, e2, ...), (d1, d2, ...)


LINE_WIDTH  = 2.5
TRAIN_COLOR = '#6BB4EF'
VAL_COLOR   = '#EFA66B'

def save_plot_loss(outpath, tloss_it, vloss_it):
    fg, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(*iter_prependaxis(tloss_it), color=TRAIN_COLOR, linewidth=LINE_WIDTH)
    ax.plot(*iter_prependaxis(vloss_it), color=VAL_COLOR,   linewidth=LINE_WIDTH)
    ax.legend(['train', 'val'])
    fg.savefig(outpath)
    plt.close(fg)


def make_plotter_loss(outdir):
    outpath = os.path.join(os.path.expanduser(outdir), 'loss.svg')
    tloss_it = []
    vloss_it = []
    def loss_plotter(epochstats):
        tloss_it.append(epochstats['train']['mean_loss'])
        vloss_it.append(epochstats['val']['mean_loss'])
        save_plot_loss(outpath, tloss_it, vloss_it)
        return {'train': tloss_it, 'val': vloss_it}
    return loss_plotter


def save_plot_recall(outpath, trecall_it, vrecall_it, mine_nth_epoch):
    fg, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall [%]')
    iter_r1   = ft.partial(map, util.first)
    iter_r100 = ft.partial(map, util.second)
    ax.plot(*iter_prependaxis(iter_r1(trecall_it), mine_nth_epoch),
            color=TRAIN_COLOR, linewidth=LINE_WIDTH)
    ax.plot(*iter_prependaxis(iter_r100(trecall_it), mine_nth_epoch),
            color=TRAIN_COLOR, linewidth=LINE_WIDTH, linestyle='dashed')
    ax.plot(*iter_prependaxis(iter_r1(vrecall_it), mine_nth_epoch),
            color=VAL_COLOR, linewidth=LINE_WIDTH)
    ax.plot(*iter_prependaxis(iter_r100(vrecall_it), mine_nth_epoch),
            color=VAL_COLOR, linewidth=LINE_WIDTH, linestyle='dashed')
    ax.legend(['train@1', 'train@100', 'val@1', 'val@100'])
    fg.savefig(outpath)
    plt.close(fg)


def is_miningepoch(epoch, mine_nth_epoch):
    return 0 == (epoch - 1)%mine_nth_epoch


def make_plotter_recall(outdir, mine_nth_epoch):
    outpath    = os.path.join(os.path.expanduser(outdir), 'recall.svg')
    trecall_it = []
    vrecall_it = []
    iter_perc = ft.partial(map, lambda x: x*100)
    def recall_plotter(stats):
        if not is_miningepoch(stats['epoch'], mine_nth_epoch):
            return {'train': trecall_it, 'val': vrecall_it}
        trecall_it.append(tuple(iter_perc(stats['train']['recall'])))
        vrecall_it.append(tuple(iter_perc(stats['val']['recall'])))
        save_plot_recall(outpath, trecall_it, vrecall_it, mine_nth_epoch)
        return {'train': trecall_it, 'val': vrecall_it}
    return recall_plotter

