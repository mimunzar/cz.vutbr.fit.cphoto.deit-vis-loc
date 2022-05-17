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


def iter_epoch_im_triplets(device, params, fn_fwd, im_it, rd_it):
    iter_im_tp = ft.partial(iter_im_triplet,
            device, params, fn_fwd, tuple(im_it), tuple(rd_it))
    return util.flatten(util.repeatedly(
        lambda: util.take(params['mine_every'], it.repeat(tuple(iter_im_tp())))))
    # => (((t1, t2, ...), (t1, t2, ...), ...),         ; epoch1
    #     ((t1, t2, ...), (t1, t2, ...), ...), ...)    ; epoch2


def iter_bwd_loss(optim, loss_it):
    loss_it = tuple(loss_it)
    for l in loss_it:
        l.backward()
    optim.step()
    optim.zero_grad()
    return loss_it


def triplet_loss(margin, fn_fwd, anchor, pos, neg):
    dist = ft.partial(cosine_dist, fn_fwd(anchor['path']))
    return torch.clamp(dist(fn_fwd(pos['path'])) - dist(fn_fwd(neg['path'])) + margin, min=0)


def im_loss(optim, margin, fn_fwd, tp_it):
    tp_it   = tuple(tp_it)
    loss_it = it.starmap(ft.partial(triplet_loss, margin, fn_fwd), tp_it)
    return sum(map(float, iter_bwd_loss(optim, loss_it)))/len(tp_it)


def iter_im_loss(optim, margin, fn_fwd, im_tp_it):
    return map(ft.partial(im_loss, optim, margin, fn_fwd), im_tp_it)


def load_im(device, fpath):
    return T.to_tensor(Image.open(fpath)).unsqueeze(0).to(device)


def iter_epoch_im_loss(model, params, im_it, rd_it):
    fwd = util.compose(model['net'], ft.partial(load_im, model['device']))
    return map(ft.partial(iter_im_loss, model['optim'], params['margin'], fwd),
        iter_epoch_im_triplets(model['device'], params, fwd, im_it, rd_it))


def make_epoch_stats(epoch_idx, im_it):
    total_ims = len(tuple(im_it))
    avg_imsec = log.make_avg_ims_sec()
    prog_bar  = log.make_progress_bar(bar_width=30, total=total_ims)
    stage     = f'Epoch {epoch_idx}'
    print(f'{prog_bar(stage, 0, 0, 0)}', end='\r', flush=True)
    def epoch_stats(acc, x):
        im_idx, loss = x
        speed   = avg_imsec(1)
        loss    = acc['loss'] + loss
        print(f'\033[K{prog_bar(stage, im_idx, speed, loss)}',
                end='\n' if total_ims == im_idx else '\r' , flush=True)
        return {'loss': loss, 'speed': speed}
    return epoch_stats


def iter_epoch_stats(model, params, im_it, rd_it):
    im_it = tuple(im_it)
    def epoch_stat(epoch_idx, im_loss_it):
        return ft.reduce(make_epoch_stats(epoch_idx, im_it),
                enumerate(im_loss_it), {'loss': 0, 'speed': 0})
    return it.starmap(epoch_stat, enumerate(iter_epoch_losses(model, params, im_it, rd_it), 1))


def iter_training(model, params, logfile, images):
    rd_it       = tuple(images['renders'])
    t_it, v_it  = map(list,  util.pluck(['train', 'val'], images))
    train_epoch = ft.partial(do_epoch, train_on_minibatch,
            lambda e: log.log(f'Training epoch {e}:\n', start='' if 1 == e else '\n', file=logfile))
    eval_epoch  = ft.partial(do_epoch, eval_minibatch,
            lambda e: log.log(f'Evaluating epoch {e}:\n', start='\n', file=logfile))
    def train_and_evaluate_epoch(e):
        random.shuffle(t_it)
        random.shuffle(v_it)
        t = train_epoch(e, model, params, logfile, t_it, rd_it)
        v = eval_epoch (e, model, params, logfile, v_it, rd_it)
        return {'epoch': e, 'train': t, 'val': v}
    return map(train_and_evaluate_epoch, it.count(1))


def make_is_learning(patience, min_delta=0.01):
    q_losses = cl.deque(maxlen=patience + 1)
    le_delta = lambda l, r: l - r <= min_delta
    def is_learning(epoch_stats):
        q_losses.append(epoch_stats['val']['loss'])
        full_queue = patience < len(q_losses)
        rest_queue = it.islice(q_losses, 1, len(q_losses))
        min_first  = all(it.starmap(le_delta, zip(it.repeat(q_losses[0]), rest_queue)))
        return not (full_queue and min_first)
    return is_learning


def on_epoch_end(logfile, callbacks_it, stats):
    print('', file=logfile, flush=True)
    util.dorun(map(lambda s: print(s, file=logfile, flush=True), log.fmt_table([
        ['',             'Train',                            'Val'],
        ['Avg. Loss',    f'{stats["train"]["loss"]:.4f}',    f'{stats["val"]["loss"]:.4f}'],
        ['Avg. Samples', f'{stats["train"]["samples"]:.4f}', f'{stats["val"]["samples"]:.4f}'],
    ])))
    util.dorun(map(lambda f: f(stats), callbacks_it))
    return stats


def train(logfile, params, model, images, callbacks_it):
    train_it = iter_training(model, params, logfile, images)
    epoch_it = it.takewhile(make_is_learning(params['patience']), util.take(params['max_epochs'], train_it))
    return min(map(ft.partial(on_epoch_end, logfile, callbacks_it), epoch_it), key=lambda e: e['val']['loss'])

