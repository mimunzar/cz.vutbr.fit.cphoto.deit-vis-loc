#!/usr/bin/env python3

import functools as ft
import itertools as it
import math as ma

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


def iter_hardnegrenders(params, im, rdbydist_it):
    return util.take(params['negatives']['samples'],
            iter_negrenders(params['negatives'], im, rdbydist_it))
    #^ Hardest negatives have shortest descriptor distances from an image.


def iter_posrenders(params, im, rd_it):
    d, d_tol = util.pluck(['dist_m', 'dist_tol_m'], params)
    y, y_tol = map(ma.radians, util.pluck(['yaw_deg','yaw_tol_deg'], params))
    return filter(ft.partial(is_dist_close, d, d_tol, im),
            filter(ft.partial(is_yaw_close, y, y_tol, im), rd_it))


def iter_hardposrenders(params, im, rdbydist_it):
    return util.take_last(params['positives']['samples'],
            iter_posrenders(params['positives'], im, rdbydist_it))
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
    rdd_it      = iter_renderdist(fn_fwd, fn_mem_fwd, rd_it, im)
    rdbydist_it = tuple(map(util.first, sorted(rdd_it, key=util.second)))
    return ((im,),
            tuple(iter_hardposrenders(params, im, rdbydist_it)),
            tuple(iter_hardnegrenders(params, im, rdbydist_it)))
    # => ((im,), (p1, p2,...), (n1, n2,...))


def iter_im_triplet(params, fn_mem_fwd, fn_fwd, rd_it, im_it):
    return it.starmap(util.compose(tuple, it.product),
        map(ft.partial(iter_imposneg, params, fn_fwd, fn_mem_fwd, tuple(rd_it)), im_it))
    # => ((t1, t2, ...),       ; im1
    #     (t1, t2, ...), ...)  ; im2


def iter_trainval_im_triplet(device, params, fn_fwd, vim_it, tim_it, rd_it):
    iter_tp = ft.partial(iter_im_triplet,
            params, util.memoize_tensor(device, fn_fwd), fn_fwd, rd_it)
    return (tuple(iter_tp(tim_it)),
            tuple(iter_tp(vim_it)))
    # => (train, val)


def iter_epoch_trainval_im_triplet(device, params, fn_fwd, vim_it, tim_it, rd_it):
    iter_trainval = ft.partial(iter_trainval_im_triplet,
            device, params, fn_fwd, tuple(vim_it), tuple(tim_it), tuple(rd_it))
    return util.flatten(util.repeatedly(
        lambda: util.take(params['mine_every_epoch'], it.repeat(iter_trainval()))))
    # => ((train, val),         ; epoch1
    #     (train, val), ...)    ; epoch2


def triplet_loss(margin, anchor, pos, neg):
    return N.triplet_margin_with_distance_loss(anchor,
            pos, neg, margin=margin, distance_function=cosine_dist)


def mean_triplet_loss(margin, fn_fwd, triplet_it):
    fwd = util.compose(fn_fwd, ft.partial(util.pluck, ['path']))
    return triplet_loss(margin, *map(
        lambda i: torch.cat(tuple(map(fwd, i))), zip(*triplet_it))).mean()


def iter_batchloss(params, fn_fwd, im_triplet_it):
    return map(ft.partial(mean_triplet_loss, params['margin'], fn_fwd),
        util.partition(params['batch_size'], util.flatten(im_triplet_it), strict=False))
    # => (l1, l2, ...)


def make_epochstat(label, params, im_triplet_it):
    total_im      = len(tuple(im_triplet_it))
    total_im_tp   = params['positives']['samples']*params['negatives']['samples']
    i, batch_size = (0, params['batch_size'])
    avg_loss      = util.make_running_avg()
    avg_imsec     = util.make_avg_ims_sec()
    prog_bar      = log.make_progress_bar(bar_width=30, total=total_im)
    print(f'{prog_bar(label, 0, 0, 0)}', end='\r', flush=True)
    def epochstat(acc, batchloss):
        nonlocal i
        i        = i + 1
        im_done  = (i*batch_size)//total_im_tp
        im_speed = avg_imsec(im_done)
        loss     = avg_loss(batchloss)
        print(f'\033[K{prog_bar(label, im_done, im_speed, loss)}',
            end='\n' if total_im == im_done else '\r', flush=True)
        return util.assoc(acc, ('avgLoss', loss), ('avgSpeed', im_speed))
    return epochstat


def val_one_epoch(params, fn_fwd, epoch, im_triplet_it):
    with torch.no_grad():
        im_triplet_it = tuple(im_triplet_it)
        return ft.reduce(make_epochstat(f'Val {epoch}', params, im_triplet_it),
                map(float, iter_batchloss(params, fn_fwd, im_triplet_it)), {})


def backward(optim, loss):
    optim.zero_grad(); loss.backward(); optim.step()
    return loss.detach()


def train_one_epoch(optim, params, fn_fwd, epoch, im_triplet_it):
    with torch.enable_grad():
        im_triplet_it = tuple(im_triplet_it)
        return ft.reduce(make_epochstat(f'Epoch {epoch}', params, im_triplet_it),
                map(util.compose(float, ft.partial(backward, optim)),
                    iter_batchloss(params, fn_fwd, im_triplet_it)), {})


def load_im(device, fpath):
    return T.to_tensor(Image.open(fpath)).unsqueeze(0).to(device)


def iter_training(model, params, vim_it, tim_it, rd_it):
    forward    = util.compose(model['net'], ft.partial(load_im, model['device']))
    iter_epoch = ft.partial(iter_epoch_trainval_im_triplet, model['device'], params, forward)
    train_one  = ft.partial(train_one_epoch, model['optim'], params, forward)
    val_one    = ft.partial(val_one_epoch, params, forward)
    def one_epoch(epoch, trainval_im_triplet_it):
        t, v = trainval_im_triplet_it
        return {'train': train_one(epoch, t), 'val': val_one(epoch, v)}
    return it.starmap(one_epoch, enumerate(iter_epoch(vim_it, tim_it, rd_it), 1))
    # => (stat1, stat2, ...)

