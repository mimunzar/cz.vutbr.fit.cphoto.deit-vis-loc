#!/usr/bin/env python3

import functools as ft
import os

import torch
import matplotlib.pyplot as plot

import src.deit_vis_loc.libs.util as util


def make_save_net(nn, prefix, output_dir, **_):
    def save_net(stats):
        epoch = f'{stats["epoch"]:03}'
        torch.save(nn.module, os.path.join(output_dir, f'{prefix}-{epoch}.torch'))
    return save_net


def add_xaxis(iterable):
    return zip(*enumerate(iterable, 1))


def make_plot_batch_loss(prefix, output_dir, **_):
    t_it  = []
    pluck = ft.partial(util.pluck, ['loss'])
    def plot_batch_loss(stats):
        t_it.extend(map(pluck, stats['train']['batches']))
        fg, ax = plot.subplots()
        ax.plot(*add_xaxis(t_it), linewidth=2.5)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        fg.savefig(os.path.join(output_dir, f'{prefix}.batch-loss.svg'))
        return t_it
    return plot_batch_loss


def make_plot_epoch_loss(prefix, output_dir, **_):
    t_it = []
    v_it = []
    def plot_epoch_loss(stats):
        t_it.append(util.pluck(['loss'], stats['train']))
        v_it.append(util.pluck(['loss'], stats['val']))
        fg, ax = plot.subplots()
        ax.plot(*add_xaxis(t_it), *add_xaxis(v_it), linewidth=2.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(['train', 'val'])
        fg.savefig(os.path.join(output_dir, f'{prefix}.epoch-loss.svg'))
        return {'train': t_it, 'val': v_it}
    return plot_epoch_loss


def make_plot_epoch_samples(prefix, output_dir, **_):
    t_it = []
    v_it = []
    def plot_epoch_samples(stats):
        t_it.append(util.pluck(['samples'], stats['train']))
        v_it.append(util.pluck(['samples'], stats['val']))
        fg, ax = plot.subplots()
        ax.plot(*add_xaxis(t_it), *add_xaxis(v_it), linewidth=2.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Samples')
        ax.legend(['train', 'val'])
        fg.savefig(os.path.join(output_dir, f'{prefix}.epoch-samples.svg'))
        return {'train': t_it, 'val': v_it}
    return plot_epoch_samples

