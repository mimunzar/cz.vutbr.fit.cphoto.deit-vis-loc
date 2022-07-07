#!/usr/bin/env python3

import functools as ft
import itertools as it
import math as ma
import statistics as st

import torch
import torch.nn.functional as N

import src.deit_vis_loc.libs.image as image
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.spherical as spherical
import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.training.callbacks as callbacks



def print_recall(trecall_it, vrecall_it):
    table_it = log.fmt_table([
        ['',           'Train',                          'Val'],
        ['Recall@1',   f'{util.first (trecall_it):.2%}', f'{util.first (vrecall_it):.2%}'],
        ['Recall@100', f'{util.second(trecall_it):.2%}', f'{util.second(vrecall_it):.2%}']])
    util.dorun(map(print, it.chain([''], table_it, [''])))


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


def cosine_dist(desc, other):
    return 1 - N.cosine_similarity(desc, other)


def iter_renderbydist(fn_iter_desc, fn_mem_iter_desc, rd_it, im):
    rd_it   = tuple(rd_it)
    stack   = util.compose(torch.stack, tuple)
    dist_it = cosine_dist(
            stack(fn_iter_desc((im,))),
            stack(fn_mem_iter_desc(rd_it)))
    return util.pluck(torch.argsort(dist_it), rd_it)
    # => (rd1, rd2, ...)


def locale1_locale100(params, im, rdbydist_it):
    rdbydist_it = tuple(rdbydist_it)
    positive_in = util.compose(
        any, ft.partial(filter, make_is_posrender(params['positives'], im)))
    return (positive_in(util.take(1,   rdbydist_it)),
            positive_in(util.take(100, rdbydist_it)))


def iter_im_localestriplets(params, fn_iter_desc, fn_mem_iter_desc, rd_it, im_it):
    iter_rdbydist = ft.partial(iter_renderbydist, fn_iter_desc, fn_mem_iter_desc)
    def localetriplets(rd_it, im):
        rdbydist_it = tuple(iter_rdbydist(rd_it, im))
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
    # => ((recall1, recall100),
    #     ((triplet1, triplet2, ...),       ; im1
    #      (triplet1, triplet2, ...), ...)  ; im2..n


def make_mem_iter_desc(fn_iter_desc, im_it):
    im_it      = tuple(im_it)
    iter_path  = ft.partial(map, ft.partial(util.pluck, ['path']))
    desc_cache = dict(zip(iter_path(im_it), fn_iter_desc(im_it)))
    def mem_iter_desc(im_it):
        yield from util.pluck(iter_path(im_it), desc_cache)
        # => (desc1, desc2, ...)
    return mem_iter_desc


def iter_desc(model, im_it):
    net, device = util.pluck(['net', 'device'], model)
    stack       = util.compose(torch.stack, tuple)
    iter_tensor = ft.partial(map, util.compose(
        image.read, ft.partial(util.pluck, ['path'])))
    def iter_batch_desc(im_it):
        return net(stack(iter_tensor(im_it)).to(device))
    return util.flatten(map(iter_batch_desc,
        util.partition(model['gpu_imcap'], im_it, strict=False)))
    # => (desc1, desc2, ...)


TRN_RECALL = None
VAL_RECALL = None

def epoch_feed(fn_iter_desc, model, params, vim_it, tim_it, rd_it):
    global TRN_RECALL, VAL_RECALL
    rd_it   = tuple(rd_it)
    rec_trp = ft.partial(recalls_imtriplets, params, fn_iter_desc)
    model['net'].eval()
    with torch.no_grad():
        mem_iter_desc          = make_mem_iter_desc(fn_iter_desc, rd_it)
        TRN_RECALL, trn_trp_it = rec_trp(mem_iter_desc, rd_it, tim_it)
        VAL_RECALL, val_trp_it = rec_trp(mem_iter_desc, rd_it, vim_it)
        print_recall(TRN_RECALL, VAL_RECALL)
        return (tuple(trn_trp_it), tuple(val_trp_it))


def iter_epoch_feed(fn_iter_desc, model, params, vim_it, tim_it, rd_it):
    mine_nth_epoch = params['mine_every_epoch']
    tv_im_triplets = ft.partial(epoch_feed,
        fn_iter_desc, model, params, tuple(vim_it), tuple(tim_it), tuple(rd_it))
    return util.flatten(util.repeatedly(
        lambda: util.take(mine_nth_epoch, it.repeat(tv_im_triplets()))))
    # => ((train, val),         ; epoch1
    #     (train, val), ...)    ; epoch2..n


def iter_im_batch_triplets(batch_size, im_triplets_it):
    def iter_batch_triplets(triplets_it):
        return util.partition(batch_size, triplets_it, strict=False)
    return map(iter_batch_triplets, im_triplets_it)
    # => (((triplet1, triplet2, ...),       ;batch1  im1
    #      (triplet1, triplet2, ...), ...), ;batch2
    #     ((triplet1, triplet2, ...),       ;batch1  im2
    #      (triplet1, triplet2, ...), ...), ;batch2
    #     ...)


def mean_triplet_loss(fn_iter_desc, margin, triplet_it):
    stack_desc = util.compose(torch.stack, tuple, fn_iter_desc)
    def triplet_loss(anchor, pos, neg):
        return N.triplet_margin_with_distance_loss(anchor,
                pos, neg, margin=margin, distance_function=cosine_dist)
    return triplet_loss(*map(stack_desc, zip(*triplet_it))).mean()


def iter_im_batchloss(fn_iter_desc, params, im_triplets_it):
    mean_loss = ft.partial(mean_triplet_loss, fn_iter_desc, params['margin'])
    return map(ft.partial(map, mean_loss),
            iter_im_batch_triplets(params['batch_size'], im_triplets_it))
    # => ((loss1, loss2, ...),          ; im1
    #     (loss1, loss2, ...), ...)     ; im2


def iter_imloss(fn_iter_desc, fn_loss_hook, params, im_triplets_it):
    loss_hook = util.compose(float, fn_loss_hook)
    return map(util.compose(st.mean, ft.partial(map, loss_hook)),
            iter_im_batchloss(fn_iter_desc, params, im_triplets_it))
    # => (imloss1, imloss2, ...)    ; epoch


def make_epochstat(label, im_triplets_it):
    total_ims  = len(tuple(im_triplets_it))
    done_ims   = 0
    mean_loss  = util.make_running_mean()
    mean_speed = util.make_mean_ims_sec()
    prog_bar   = log.make_progress_bar(30, total_ims)
    print(f'{prog_bar(label, 0, 0, 0)}', end='\r')
    def epochstat(acc, imloss):
        nonlocal done_ims
        done_ims = done_ims + 1
        loss     = mean_loss(imloss)
        speed    = mean_speed(done_ims)
        print(f'\033[K{prog_bar(label, done_ims, speed, loss)}',
                end='\n' if total_ims == done_ims else '\r')
        return util.assoc(acc, ('mean_loss',  loss), ('mean_speed', speed))
    return epochstat


def val_one_epoch(fn_iter_desc, model, params, epoch, im_triplets_it):
    im_triplets_it = tuple(im_triplets_it)
    epochstat      = make_epochstat(f'Val {epoch}', im_triplets_it)
    imloss_it      = iter_imloss(fn_iter_desc, util.identity, params, im_triplets_it)
    model['net'].eval()
    with torch.no_grad():
        return ft.reduce(epochstat, imloss_it, {'recall': VAL_RECALL})


def backward(optim, loss):
    optim.zero_grad(); loss.backward(); optim.step()
    return loss.detach()


def train_one_epoch(fn_iter_desc, model, params, epoch, im_triplets_it):
    im_triplets_it = tuple(im_triplets_it)
    epochstat      = make_epochstat(f'Epoch {epoch}', im_triplets_it)
    imloss_it      = iter_imloss(fn_iter_desc,
            ft.partial(backward, model['optim']), params, im_triplets_it)
    model['net'].train()
    with torch.enable_grad():
        return ft.reduce(epochstat, imloss_it, {'recall': TRN_RECALL})


def iter_trainingepoch(model, params, vim_it, tim_it, rd_it):
    iter_dsc   = ft.partial(iter_desc, model)
    iter_epoch = ft.partial(iter_epoch_feed, iter_dsc, model, params)
    train_one  = ft.partial(train_one_epoch, iter_dsc, model, params)
    val_one    = ft.partial(val_one_epoch,   iter_dsc, model, params)
    def one_epoch(epoch, epoch_feed):
        t_stat = train_one(epoch, util.first (epoch_feed))
        v_stat = val_one  (epoch, util.second(epoch_feed))
        model['scheduler'].step()
        return {'epoch': epoch, 'train': t_stat, 'val': v_stat}
    return it.starmap(one_epoch, enumerate(iter_epoch(vim_it, tim_it, rd_it), 1))
    # => (epochstat1, epochstat2, ...)


def train(model, params, output_dir, vim_it, tim_it, rd_it):
    on_epoch = util.juxt(
        callbacks.make_netsaver(output_dir, model['net']),
        callbacks.make_loss_plotter(output_dir),
        callbacks.make_recall_plotter(output_dir, params['mine_every_epoch']))
    util.dorun(
        util.take(params['max_epochs'],
            map(on_epoch, iter_trainingepoch(model, params, vim_it, tim_it, rd_it))))

