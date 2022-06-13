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



def print_recall(trecall_it, vrecall_it):
    content_it = log.fmt_table([
        ['',           'Train',                          'Val'],
        ['Recall@1',   f'{util.first (trecall_it):.2%}', f'{util.first (vrecall_it):.2%}'],
        ['Recall@100', f'{util.second(trecall_it):.2%}', f'{util.second(vrecall_it):.2%}']])
    util.dorun(map(print, it.chain([''], content_it, [''])))


def is_dist_close(limit_m, tol_m, im, render):
    latlon = ft.partial(util.pluck, ['latitude', 'longitude'])
    return spherical.dist_m(latlon(im), latlon(render)) - limit_m <= tol_m


def iter_negrenders(params, im, rd_it):
    d, d_tol = util.pluck(['dist_m', 'dist_tol_m'], params)
    return filter(util.complement(ft.partial(is_dist_close, d, d_tol, im)), rd_it)
    #^ Renders are considered as negatives only when they lie distant after a
    # certain threshold. This forces the method to learn local features rather
    # than distant landmarks. This is different from how humans do it.


def iter_hardnegrenders(params, im, rdbydist_it):
    return util.take(params['negatives']['samples'],
            iter_negrenders(params['negatives'], im, rdbydist_it))
    #^ Take hard negatives with low descriptor distance from an image. A hard
    # negative is ordered closer to its query image. We want to avoid for the
    # network to recall any negatives before a positive.


def is_yaw_close(limit_rad, tol_rad, im, render):
    yaw = ft.partial(util.pluck, ['yaw'])
    return spherical.circle_dist_rad(yaw(im), yaw(render)) - limit_rad <= tol_rad


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

def iter_renderdist(fn_iter_desc, fn_mem_iter_desc, rd_it, im):
    stack = util.compose(torch.stack, tuple)
    dist  = ft.partial(cosine_dist, stack(fn_iter_desc((im,))))
    def iter_batch_renderdist(rd_it):
        rd_it = tuple(rd_it)
        return zip(rd_it, dist(stack(fn_mem_iter_desc(rd_it))))
    return util.flatten(map(iter_batch_renderdist,
        util.partition(DIST_PART_SIZE, rd_it, strict=False)))
    # => ((render1, dist1), (render2, dist2), ...)


def locale1_locale100(params, im, rdbydist_it):
    rdbydist_it = tuple(rdbydist_it)
    positive_in = util.compose(
        any, ft.partial(filter, make_is_posrender(params['positives'], im)))
    return (positive_in(util.take(1,   rdbydist_it)),
            positive_in(util.take(100, rdbydist_it)))
    # => (locale1, locale100)


def iter_im_localestriplets(params, fn_iter_desc, fn_mem_iter_desc, rd_it, im_it):
    iter_rendersbyimdist = util.compose(
        ft.partial(map, util.first),
        ft.partial(util.sortby, util.second),
        ft.partial(iter_renderdist, fn_iter_desc, fn_mem_iter_desc))
    def localetriplets(rd_it, im):
        rdbydist_it = tuple(iter_rendersbyimdist(rd_it, im))
        triplet_it  = iter_imtriplets(params, im, rdbydist_it)
        return (locale1_locale100(params, im, rdbydist_it), tuple(triplet_it))
    return map(ft.partial(localetriplets, tuple(rd_it)), im_it)
    # => (((locale1, locale2, ...), (triplet1, triplet2, ...)),       ; im1
    #     ((locale1, locale2, ...), (triplet1, triplet2, ...)), ...)  ; im2..n


def recalls_imtriplets(params, fn_iter_desc, fn_mem_iter_desc, rd_it, im_it):
    im_locales_it, im_triplets_it = zip(*
        iter_im_localestriplets(params, fn_iter_desc, fn_mem_iter_desc, rd_it, im_it))
    def locale_to_recall(locale_it):
        locale_it = tuple(locale_it)
        return sum(locale_it)/len(locale_it)
    return (tuple(map(locale_to_recall, zip(*im_locales_it))), im_triplets_it)
    # => ((recall1, recall100), ((triplet1, triplet2, ...), (triplet1, triplet2, ...), ...)
    #                            ^im1                       ^im2


def make_mem_iter_desc(fn_iter_desc, im_it):
    im_it      = tuple(im_it)
    iter_path  = ft.partial(map, ft.partial(util.pluck, ['path']))
    desc_cache = dict(zip(iter_path(im_it), fn_iter_desc(im_it)))
    def mem_iter_desc(im_it):
        yield from util.pluck(iter_path(im_it), desc_cache)
        # => (desc1, desc2, ...)
    return mem_iter_desc


def iter_desc(gpu_imcap, model, im_it):
    net, device = util.pluck(['net', 'device'], model)
    stack       = util.compose(torch.stack, tuple)
    iter_tensor = ft.partial(map, util.compose(
        T.to_tensor, Image.open, ft.partial(util.pluck, ['path'])))
    def iter_batch_desc(im_it):
        return net(stack(iter_tensor(im_it)).to(device))
    return util.flatten(map(iter_batch_desc,
        util.partition(gpu_imcap, im_it, strict=False)))
    # => (desc1, desc2, ...)


TRN_RECALL = None
VAL_RECALL = None

def epoch_feed(model, params, vim_it, tim_it, rd_it):
    global TRN_RECALL, VAL_RECALL
    rd_it = tuple(rd_it)
    with torch.no_grad():
        model['net'].eval()
        iter_dsc = ft.partial(iter_desc, params['gpu_imcap'], model)
        rcl_trp  = ft.partial(recalls_imtriplets, params,
                iter_dsc, make_mem_iter_desc(iter_dsc, rd_it), rd_it)
        TRN_RECALL, trn_trp_it = rcl_trp(tim_it)
        VAL_RECALL, val_trp_it = rcl_trp(vim_it)
        print_recall(TRN_RECALL, VAL_RECALL)
        return (tuple(trn_trp_it), tuple(val_trp_it))


def iter_epoch_feed(model, params, vim_it, tim_it, rd_it):
    triplets = ft.partial(epoch_feed, model,
            params, tuple(vim_it), tuple(tim_it), tuple(rd_it))
    return util.flatten(util.repeatedly(
        lambda: util.take(params['mine_every_epoch'], it.repeat(triplets()))))
    # => ((train, val),         ; epoch1
    #     (train, val), ...)    ; epoch2


def triplet_loss(margin, anchor, pos, neg):
    return N.triplet_margin_with_distance_loss(anchor,
            pos, neg, margin=margin, distance_function=cosine_dist)


def mean_triplet_loss(margin, fn_iter_desc, triplet_it):
    stack_desc = util.compose(torch.stack, tuple, fn_iter_desc)
    return triplet_loss(margin, *map(stack_desc, zip(*triplet_it))).mean()


def iter_batchloss(params, fn_iter_desc, im_triplet_it):
    return map(ft.partial(mean_triplet_loss, params['margin'], fn_iter_desc),
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


def val_one_epoch(model, params, epoch, im_triplet_it):
    im_triplet_it = tuple(im_triplet_it)
    iter_dsc      = ft.partial(iter_desc, params['gpu_imcap'], model)
    with torch.no_grad():
        model['net'].eval()
        return ft.reduce(make_epochstat(f'Val {epoch}', params, im_triplet_it),
                map(float, iter_batchloss(params, iter_dsc, im_triplet_it)),
                {'recall': VAL_RECALL})


def backward(optim, loss):
    optim.zero_grad(); loss.backward(); optim.step()
    return loss.detach()


def train_one_epoch(model, params, epoch, im_triplet_it):
    im_triplet_it = tuple(im_triplet_it)
    iter_dsc      = ft.partial(iter_desc, params['gpu_imcap'], model)
    with torch.enable_grad():
        model['net'].train()
        return ft.reduce(make_epochstat(f'Epoch {epoch}', params, im_triplet_it),
                map(util.compose(float, ft.partial(backward, model['optim'])),
                    iter_batchloss(params, iter_dsc, im_triplet_it)),
                {'recall': TRN_RECALL})


def iter_trainingepoch(model, params, vim_it, tim_it, rd_it):
    iter_epoch = ft.partial(iter_epoch_feed, model, params)
    train_one  = ft.partial(train_one_epoch, model, params)
    val_one    = ft.partial(val_one_epoch,   model, params)
    def one_epoch(epoch, trainval_im_triplet_it):
        t, v = trainval_im_triplet_it
        return {'epoch': epoch, 'train': train_one(epoch, t), 'val': val_one(epoch, v)}
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

