#!/usr/bin/env python3

import collections as cl
import functools as ft
import itertools as it
import math as ma
import random
import statistics as st

import torch
import torch.nn.functional as N
import torchvision.transforms.functional as T
from PIL import Image

import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.spherical as spherical


def is_dist_close(limit_m, tol_m, im, render):
    pluck = ft.partial(util.pluck, ['latitude', 'longitude'])
    return spherical.dist_m(pluck(im), pluck(render)) - limit_m <= tol_m


def is_yaw_close(limit_rad, tol_rad, im, render):
    pluck = ft.partial(util.pluck, ['yaw'])
    return spherical.circle_dist_rad(pluck(im), pluck(render)) - limit_rad <= tol_rad


def iter_negrenders(params, im, rd_it):
    d, d_tol = util.pluck(['dist_m', 'dist_tol_m'], params)
    return filter(util.complement(ft.partial(is_dist_close, d, d_tol, im)), rd_it)


def iter_hardnegrenders(params, im, rd_by_dist_it):
    return util.take(params['negatives']['samples'],
            iter_negrenders(params['negatives'], im, rd_by_dist_it))
    #^ Hardest negatives have shortest descriptor distances from an image.


def iter_posrenders(params, im, rd_it):
    d, d_tol = util.pluck(['dist_m', 'dist_tol_m'], params)
    y, y_tol = map(ma.radians, util.pluck(['yaw_deg','yaw_tol_deg'], params))
    return filter(ft.partial(is_dist_close, d, d_tol, im),
            filter(ft.partial(is_yaw_close, y, y_tol, im), rd_it))


def iter_hardposrenders(params, im, rd_by_dist_it):
    return util.take_last(params['positives']['samples'],
            iter_posrenders(params['positives'], im, rd_by_dist_it))
    #^ Hardest positives have longest descriptor distances from an image.


def cosine_dist(emb, emb_other):
    return 1 - N.cosine_similarity(emb, emb_other)


def iter_renderdist(fn_fwd, fn_mem_fwd, rd_it, im):
    path = ft.partial(util.pluck, ['path'])
    dist = ft.partial(cosine_dist, fn_fwd(path(im)))
    def iter_im_render_batch_dist(rd_it):
        with torch.no_grad():
            e_it = map(util.compose(fn_mem_fwd, path), rd_it)
            return zip(rd_it, dist(torch.cat(tuple(e_it))).cpu())
    return util.flatten(
            map(iter_im_render_batch_dist, util.partition(1000, rd_it, strict=False)))
    # => ((r1, d1), (r2, d2), ...)


def iter_imposneg(params, fn_fwd, fn_mem_fwd, rd_it, im):
    rdd_it     = iter_renderdist(fn_fwd, fn_mem_fwd, rd_it, im)
    by_dist_it = tuple(map(util.first, sorted(rdd_it, key=util.second)))
    return ((im,),
            tuple(iter_hardposrenders(params, im, by_dist_it)),
            tuple(iter_hardnegrenders(params, im, by_dist_it)))
    # => ((im,), (pr1, pr2,...), (nr1, nr2,...))


def iter_im_triplet(device, params, fn_fwd, im_it, rd_it):
    mem_fwd = util.memoize_tensor(device, fn_fwd)
    return it.starmap(util.compose(tuple, it.product),
        map(ft.partial(iter_imposneg, params, fn_fwd, mem_fwd, tuple(rd_it)), im_it))
    # => ((t1, t2, ...),       ; im1
    #     (t1, t2, ...), ...)  ; im2


def iter_epoch_im_triplet(device, params, fn_fwd, im_it, rd_it):
    iter_im_tp = ft.partial(iter_im_triplet,
            device, params, fn_fwd, tuple(im_it), tuple(rd_it))
    return util.flatten(util.repeatedly(
        lambda: util.take(params['mine_every_epoch'], it.repeat(tuple(iter_im_tp())))))
    # => (((t1, t2, ...), (t1, t2, ...), ...),         ; epoch1
    #     ((t1, t2, ...), (t1, t2, ...), ...), ...)    ; epoch2


def triplet_loss(margin, anchor, pos, neg):
    return N.triplet_margin_with_distance_loss(anchor, pos, neg,
            margin=margin, distance_function=cosine_dist, reduction='mean')


def batchloss(margin, fn_fwd, triplet_it):
    fwd = util.compose(fn_fwd, ft.partial(util.pluck, ['path']))
    return triplet_loss(margin, *map(
        lambda i: torch.cat(tuple(map(fwd, i))), zip(*triplet_it)))


def backward(optim, loss):
    optim.zero_grad(); loss.backward(); optim.step()
    return loss.detach()


def imloss(optim, params, fn_fwd, triplet_it):
    return sum(map(util.compose(float, ft.partial(backward, optim)),
                map(ft.partial(batchloss, params['margin'], fn_fwd),
                    util.partition(params['batch_size'], triplet_it, strict=False))))


def make_epochstat(desc, im_triplet_it):
    total_ims = len(tuple(im_triplet_it))
    avg_loss  = util.make_running_avg()
    avg_imsec = log.make_avg_ims_sec()
    prog_bar  = log.make_progress_bar(bar_width=30, total=total_ims)
    print(f'{prog_bar(desc, 0, 0, 0)}', end='\r', flush=True)
    def epochstat(acc, x):
        i, loss = x
        speed   = avg_imsec(1)
        loss    = avg_loss(loss)
        print(f'\033[K{prog_bar(desc, i, speed, loss)}',
                end='\n' if total_ims == i else '\r' , flush=True)
        return util.assoc(acc, ('avgLoss', loss), ('avgSpeed', speed))
    return epochstat


def epochstat(optim, params, fn_fwd, epoch, im_triplet_it):
    im_triplet_it = tuple(im_triplet_it)
    loss_it       = map(ft.partial(imloss, optim, params, fn_fwd), im_triplet_it)
    return ft.reduce(make_epochstat(f'Epoch {epoch}', im_triplet_it), enumerate(loss_it, 1), {})


def load_im(device, fpath):
    return T.to_tensor(Image.open(fpath)).unsqueeze(0).to(device)


def train(model, params, im_it, rd_it):
    fwd = util.compose(model['net'], ft.partial(load_im, model['device']))
    return util.take(params['max_epochs'],
            it.starmap(ft.partial(epochstat, model['optim'], params, fwd),
                enumerate(iter_epoch_im_triplet(model['device'], params, fwd, im_it, rd_it), 1)))
    # => (stat1, stat2, ...)

