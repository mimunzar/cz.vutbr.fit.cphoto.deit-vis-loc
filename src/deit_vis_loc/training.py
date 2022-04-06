#!/usr/bin/env python3

import collections as cl
import functools as ft
import itertools as it
import math as ma
import random
import statistics as st

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.libs.log as log
import src.deit_vis_loc.libs.spherical as spherical



def is_dist_close(fn_dist_m, limit_m, tol_m, im, render):
    pluck = ft.partial(util.pluck, ['latitude', 'longitude'])
    return fn_dist_m(pluck(im), pluck(render)) - limit_m <= tol_m


def is_yaw_close(fn_circle_dist_rad, limit_rad, tol_rad, im, render):
    pluck = ft.partial(util.pluck, ['yaw'])
    return fn_circle_dist_rad(pluck(im), pluck(render)) - limit_rad <= tol_rad


def iter_pos_renders(params, im, renders_it):
    d, d_tol  = util.pluck(['dist_m', 'dist_tolerance_m'], params)
    y, y_tol  = map(ma.radians, util.pluck(['yaw_deg','yaw_tolerance_deg'], params))
    is_d_near = ft.partial(is_dist_close, spherical.dist_m, d, d_tol, im)
    is_y_near = ft.partial(is_yaw_close, spherical.circle_dist_rad, y, y_tol, im)
    return filter(is_d_near, filter(is_y_near, renders_it))


def iter_neg_renders(params, im, renders_it):
    d, d_tol  = util.pluck(['dist_m', 'dist_tolerance_m'], params)
    is_d_near = ft.partial(is_dist_close, spherical.dist_m, d, d_tol, im)
    return filter(util.complement(is_d_near), renders_it)


def iter_triplets(fn_iter_pos, fn_iter_neg, im_it, renders_it):
    renders_it = tuple(renders_it)
    def iter_triplets(im):
        im_pos = fn_iter_pos(im, renders_it)
        im_neg = fn_iter_neg(im, renders_it)
        return it.product((im,), im_pos, im_neg)
    return map(iter_triplets, im_it)


def mining_stats(acc, mining_stats):
    return {
        'triplets': [*acc['triplets'], util.nth(2, mining_stats)],
        'samples' : util.second(mining_stats),
    }


def iter_mined_triplets(n, fn_iter_tps, fn_tp_loss, im_it, renders_it):
    prepend_loss = lambda x: (float(fn_tp_loss(*x[1])), *x)
    def choose_hard(tps_it):
        with torch.no_grad():
            samp_it = enumerate(util.rand_sample(0.1, tps_it), 1)
            hard_it = util.take(n, filter(util.first, map(prepend_loss, samp_it)))
            return ft.reduce(mining_stats, hard_it, {'samples': 0, 'triplets': []})
    return map(choose_hard, fn_iter_tps(im_it, renders_it))


def triplet_loss(margin, fn_fwd, anchor, pos, neg):
    a_embed = fn_fwd(anchor['path'])
    a_p_dis = 1 - F.cosine_similarity(a_embed, fn_fwd(pos['path']))
    a_n_dis = 1 - F.cosine_similarity(a_embed, fn_fwd(neg['path']))
    return torch.clamp(a_p_dis - a_n_dis + margin, min=0)


def iter_triplet_loss(fn_fwd, params, im_it, renders_it):
    loss     = ft.partial(triplet_loss, params['margin'], fn_fwd)
    mined_it = iter_mined_triplets(params['n_triplets'],
        ft.partial(iter_triplets,
            ft.partial(iter_pos_renders, params['positives']),
            ft.partial(iter_neg_renders, params['negatives'])), loss, im_it, renders_it)
    return map(lambda m: {**m, 'loss': sum(it.starmap(loss, m['triplets']))}, mined_it)


def make_load_im(device, input_size):
    to_tensor = T.Compose([
        T.Resize(input_size, interpolation=3),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return lambda fpath: to_tensor(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)


def make_minibatch_stats(stage, logfile, im_it):
    total_ims = len(im_it)
    avg_imsec = log.make_avg_ims_sec()
    prog_bar  = log.make_progress_bar(bar_width=30, total=total_ims)
    print(f'{prog_bar(stage, 0, 0, 0)}', end='\r', file=logfile, flush=True)
    def minibatch_stats(acc, x):
        i, mined_stats = x
        speed = avg_imsec(1)
        loss  = acc['loss'] + mined_stats['loss']
        samp  = [*acc['samples'], mined_stats['samples']]
        last  = i == total_ims
        print(f'\033[K{prog_bar(stage, i, speed, float(loss))}',
                end='\n' if last else '\r' , file=logfile, flush=True)
        return {'loss': loss, 'speed': speed, 'samples': st.median(samp) if last else samp}
    return minibatch_stats


def minibatch_stats(model, params, logfile, renders_it, batches_total, batch_id, im_it):
    im_it = tuple(im_it)
    zero  = torch.zeros(1, device=model['device'], requires_grad=torch.is_grad_enabled())
    fwd   = util.compose(model['net'], make_load_im(model['device'], params['input_size']))
    ls_it = enumerate(iter_triplet_loss(util.memoize(fwd), params, im_it, renders_it), 1)
    stats = make_minibatch_stats(f'Batch {log.fmt_fraction(batch_id, batches_total)}', logfile, im_it)
    return ft.reduce(stats, ls_it, {'loss': zero, 'speed': 0, 'samples': []})


def backward(optim, loss):
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


def train_on_minibatch(model, *args):
    with torch.enable_grad():
        model['net'].train()
        bwd = util.compose(float, ft.partial(backward, model['optim']))
        return util.update('loss', bwd, minibatch_stats(model, *args))


def eval_minibatch(model, *args):
    with torch.no_grad():
        model['net'].eval()
        return util.update('loss', float, minibatch_stats(model, *args))


def make_epoch_stats():
    ravg_loss = util.make_running_avg()
    def epoch_stats(acc, batch_stats):
        loss = ravg_loss(batch_stats['loss'])
        return {'avg_loss': loss, 'batches': [*acc['batches'], batch_stats]}
    return epoch_stats


def do_epoch(fn_batch_apply, fn_onstart, epoch, model, params, logfile, im_it, renders_it):
    fn_onstart(epoch)
    minib_it   = tuple(enumerate(util.partition(params['batch_size'], im_it), 1))
    app_minib  = ft.partial(fn_batch_apply, model, params, logfile, renders_it, len(minib_it))
    return ft.reduce(make_epoch_stats(), it.starmap(app_minib, minib_it), {'avg_loss': 0, 'batches': []})


def iter_training(model, params, logfile, images, renders_it):
    renders_it  = tuple(renders_it)
    t_it, v_it  = map(list, util.pluck(['train', 'val'], images))
    train_epoch = ft.partial(do_epoch, train_on_minibatch, lambda e: log.log(f'Training epoch {e}:\n', start='\n'))
    eval_epoch  = ft.partial(do_epoch, eval_minibatch,     lambda e: log.log(f'Evaluating epoch {e}:\n', start='\n'))
    def train_and_evaluate_epoch(epoch):
        random.shuffle(t_it)
        random.shuffle(v_it)
        t = train_epoch(epoch, model, params, logfile, t_it, renders_it)
        v = eval_epoch (epoch, model, params, logfile, v_it, renders_it)
        return {'epoch': epoch, 'train': t, 'val': v}
    return map(train_and_evaluate_epoch, it.count(1))


def make_is_learning(patience, min_delta):
    q_losses = cl.deque(maxlen=patience + 1)
    le_delta = lambda l, r: l - r <= min_delta
    def is_training(train_stats):
        q_losses.append(train_stats['vloss'])
        full_queue = patience < len(q_losses)
        rest_queue = it.islice(q_losses, 1, len(q_losses))
        min_first  = all(it.starmap(le_delta, zip(it.repeat(q_losses[0]), rest_queue)))
        return not (full_queue and min_first)
    return is_training


def train_stats(model, logfile, epoch, tloss, vloss, tspeed, vspeed, **rest):
    log.log(f'Epoch {epoch} ended, tloss: {tloss:.2f}, vloss: {vloss:.2f}, '
           + f'tspeed {tspeed:.2f}, vspeed {vspeed:.2f} im/s\n\n', start='\n', file=logfile)
    model['save_net'](epoch)
    return {**{'epoch': epoch, 'tloss': tloss, 'vloss': vloss}, **rest}


def train(model, train_params, logfile, meta, images):
    is_learning = make_is_learning(train_params['stopping_patience'], min_delta=0.01)
    training_it = iter_training(model, train_params, logfile, meta, images)
    learning_it = it.takewhile(is_learning, util.take(train_params['max_epochs'], training_it))
    return min(map(lambda d: train_stats(model, logfile, **d), learning_it), key=ft.partial(util.pluck, ['vloss']))

