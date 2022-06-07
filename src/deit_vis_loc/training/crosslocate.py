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
import src.deit_vis_loc.training.callbacks as callbacks



def has_positive_in(fn_is_pos, iterable, n):
    return any(filter(fn_is_pos, util.take(n, iterable)))


def locale1_locale100(params, im, rdbydist_it):
    positive_in = ft.partial(has_positive_in,
        make_is_posrender(params['positives'], im), tuple(rdbydist_it))
    return (positive_in(1), positive_in(100))


def is_dist_close(limit_m, tol_m, im, render):
    pluck = ft.partial(util.pluck, ['latitude', 'longitude'])
    return spherical.dist_m(pluck(im), pluck(render)) - limit_m <= tol_m


def iter_negrenders(params, im, rd_it):
    d, d_tol = util.pluck(['dist_m', 'dist_tol_m'], params)
    return filter(util.complement(ft.partial(is_dist_close, d, d_tol, im)), rd_it)


def iter_hardnegrenders(params, im, rdbydist_it):
    return util.take(params['negatives']['samples'],
            iter_negrenders(params['negatives'], im, rdbydist_it))
    #^ Take hard negatives with low descriptor distance from an image. A hard
    # negative is ordered closer to its query image. We want to avoid for the
    # network to recall any negatives before a positive.


def is_yaw_close(limit_rad, tol_rad, im, render):
    pluck = ft.partial(util.pluck, ['yaw'])
    return spherical.circle_dist_rad(pluck(im), pluck(render)) - limit_rad <= tol_rad


def make_is_posrender(params, im):
    dist_close = ft.partial(is_dist_close, *util.pluck(['dist_m', 'dist_tol_m'], params))
    yaw_close  = ft.partial(is_yaw_close,  *map(
        ma.radians, util.pluck(['yaw_deg','yaw_tol_deg'], params)))
    return lambda render: yaw_close(im, render) and dist_close(im, render)


def iter_easyposrenders(params, im, rdbydist_it):
    return util.take(params['positives']['samples'],
            filter(make_is_posrender(params['positives'], im), rdbydist_it))
    #^ Take easy positives with low descriptor distances from an image. A hard
    # positive may not resemble its query image due to various obstacles or
    # faulty GPS information.


def iter_imtriplets(params, im, rdbydist_it):
    return it.product((im,),
        iter_easyposrenders(params, im, rdbydist_it),
        iter_hardnegrenders(params, im, rdbydist_it))


def cosine_dist(emb, emb_other):
    return 1 - N.cosine_similarity(emb, emb_other)


DIST_PART_SIZE = 1000

def iter_renderdist(fn_fwd, rd_it, im):
    path = ft.partial(util.pluck, ['path'])
    dist = ft.partial(cosine_dist, fn_fwd(path(im)))
    def iter_batch_renderdist(rd_it):
        with torch.no_grad():
            e_it = map(util.compose(fn_fwd, path), rd_it)
            return zip(rd_it, dist(torch.cat(tuple(e_it))).cpu())
    return util.flatten(map(iter_batch_renderdist,
        util.partition(DIST_PART_SIZE, rd_it, strict=False)))
    # => ((render1, dist1), (render2, dist2), ...)


def iter_im_localestriplets(params, fn_fwd, rd_it, im_it):
    def localetriplets(rd_it, im):
        rdbydist_it = tuple(map(util.first,
            sorted(iter_renderdist(fn_fwd, rd_it, im), key=util.second)))
        triplet_it  = iter_imtriplets(params, im, rdbydist_it)
        return (locale1_locale100(params, im, rdbydist_it), tuple(triplet_it))
    return map(ft.partial(localetriplets, tuple(rd_it)), im_it)
    # => (((locale1, locale2, ...), (triplet1, triplet2, ...)),       ; im1
    #     ((locale1, locale2, ...), (triplet1, triplet2, ...)), ...)  ; im2..n


def recalls_triplets(params, fn_fwd, rd_it, im_it):
    l_it, t_it = zip(*iter_im_localestriplets(params, fn_fwd, rd_it, im_it))
    def iter_localerecall(im_locales_it):
        im_locales_it = tuple(im_locales_it)
        total_ims     = len(im_locales_it)
        return map(lambda recall_it: sum(recall_it)/total_ims, zip(*im_locales_it))
    return (dict(zip(('at_1', 'at_100'), iter_localerecall(l_it))), t_it)
    # => (recall, ((triplet1, triplet2, ...), (triplet1, triplet2, ...), ...)
    #              ^im1                       ^im2


TRAIN_RECALL = None
VAL_RECALL   = None

def traintriplets_valtriplets(device, params, fn_fwd, vim_it, tim_it, rd_it):
    global TRAIN_RECALL, VAL_RECALL
    recalls_triplets_of = ft.partial(recalls_triplets,
            params, util.memoize_tensor(device, fn_fwd), tuple(rd_it))
    TRAIN_RECALL, tt_it = recalls_triplets_of(tim_it)
    VAL_RECALL,   vt_it = recalls_triplets_of(vim_it)
    print()
    util.dorun(map(print, log.fmt_table([
        ['',           'Train',                         'Val'],
        ['Recall@1',   f'{TRAIN_RECALL["at_1"]:.2%}',   f'{VAL_RECALL["at_1"]:.2%}'],
        ['Recall@100', f'{TRAIN_RECALL["at_100"]:.2%}', f'{VAL_RECALL["at_100"]:.2%}']])))
    print()
    return (tuple(tt_it), tuple(vt_it))


def iter_epoch_traintriplets_valtriplets(device, params, fn_fwd, vim_it, tim_it, rd_it):
    triplets = ft.partial(traintriplets_valtriplets,
            device, params, fn_fwd, tuple(vim_it), tuple(tim_it), tuple(rd_it))
    return util.flatten(util.repeatedly(
        lambda: util.take(params['mine_every_epoch'], it.repeat(triplets()))))
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
    # => (loss1, loss2, ...)


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
        return util.assoc(acc, ('avg_loss', loss), ('avg_speed', im_speed))
    return epochstat


def val_one_epoch(params, fn_fwd, epoch, im_triplet_it):
    with torch.no_grad():
        im_triplet_it = tuple(im_triplet_it)
        return ft.reduce(make_epochstat(f'Val {epoch}', params, im_triplet_it),
                map(float, iter_batchloss(params, fn_fwd, im_triplet_it)), {'recall': VAL_RECALL})


def backward(optim, loss):
    optim.zero_grad(); loss.backward(); optim.step()
    return loss.detach()


def train_one_epoch(optim, params, fn_fwd, epoch, im_triplet_it):
    with torch.enable_grad():
        im_triplet_it = tuple(im_triplet_it)
        return ft.reduce(make_epochstat(f'Epoch {epoch}', params, im_triplet_it),
                map(util.compose(float, ft.partial(backward, optim)),
                    iter_batchloss(params, fn_fwd, im_triplet_it)), {'recall': TRAIN_RECALL})


def load_im(device, fpath):
    return T.to_tensor(Image.open(fpath)).unsqueeze(0).to(device)


def iter_trainingepoch(model, params, vim_it, tim_it, rd_it):
    forward    = util.compose(model['net'], ft.partial(load_im, model['device']))
    iter_epoch = ft.partial(iter_epoch_traintriplets_valtriplets, model['device'], params, forward)
    def one_epoch(epoch, trainval_im_triplet_it):
        t, v = trainval_im_triplet_it
        return {'epoch' : epoch,
                'train' : train_one_epoch(model['optim'], params, forward, epoch, t),
                'val'   : val_one_epoch(params, forward, epoch, v)}
    return it.starmap(one_epoch, enumerate(iter_epoch(vim_it, tim_it, rd_it), 1))
    # => (stat1, stat2, ...)


def train(model, params, output_dir, vim_it, tim_it, rd_it):
    on_epoch = util.juxt(
        callbacks.make_netsaver(output_dir, model['net']),
        callbacks.make_loss_plotter(output_dir),
        callbacks.make_recall_plotter(output_dir, params['mine_every_epoch']))
    util.dorun(
        util.take(params['max_epochs'],
            map(on_epoch, iter_trainingepoch(model, params, vim_it, tim_it, rd_it))))

