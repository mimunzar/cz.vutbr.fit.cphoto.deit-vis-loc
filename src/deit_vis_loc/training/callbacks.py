#!/usr/bin/env python3

import os

import torch
import matplotlib.pyplot as plt


def make_netsaver(output_dir, net):
    output_dir = os.path.expanduser(output_dir)
    net        = net.module if hasattr(net, 'module') else net
    def netsaver(epochstats):
        epoch = f'{epochstats["epoch"]:03}'
        torch.save(net, os.path.join(output_dir, f'net-{epoch}.torch'))
    return netsaver


def epochaxis(iterable):
    return zip(*enumerate(iterable, 1))


def make_loss_plotter(output_dir):
    output_dir = os.path.expanduser(output_dir)
    tloss_it   = []
    vloss_it   = []
    def loss_plotter(stats):
        tloss_it.append(stats['train']['avg_loss'])
        vloss_it.append(stats['val']['avg_loss'])

        fg, ax = plt.subplots()
        ax.plot(*epochaxis(tloss_it), *epochaxis(vloss_it), linewidth=2.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(['train', 'val'])
        fg.savefig(os.path.join(output_dir, f'loss.svg'))
        plt.close(fg)
        return {'train': tloss_it, 'val': vloss_it}
    return loss_plotter

