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
import src.deit_vis_loc.libs.logging as logging
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


def backward(optim, loss):
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


def make_minibatch_stats(stage, logfile, im_it):
    total_ims = len(im_it)
    avg_imsec = logging.make_avg_ims_sec()
    prog_bar  = logging.make_progress_bar(bar_width=30, total=total_ims)
    print(f'{prog_bar(stage, 0, 0, 0)}', end='\r', file=logfile, flush=True)
    def minibatch_stats(acc, x):
        i, mined_stats = x
        speed = avg_imsec(1)
        loss  = acc['loss'] + mined_stats['loss']
        end   = '\n' if i == total_ims else '\r'
        print(f'\033[K{prog_bar(stage, i, speed, float(loss))}', end=end, file=logfile, flush=True)
        return {'loss': loss, 'speed': speed, 'samples': [*acc['samples'], mined_stats['samples']]}
    return minibatch_stats


def train_minibatch(model, params, logfile, renders_it, batches_total, batch_id, im_it):
    im_it = tuple(im_it)
    bwd   = util.compose(float, ft.partial(backward, model['optim']))
    fwd   = util.compose(model['net'], make_load_im(model['device'], params['input_size']))
    ms_it = iter_triplet_loss(util.memoize(fwd), params, im_it, renders_it)
    stats = make_minibatch_stats(f'Batch {logging.format_fraction(batch_id, batches_total)}', logfile, im_it)
    with torch.enable_grad():
        model['net'].train()
        zero = torch.zeros(1, device=model['device'], requires_grad=True)
        stat = ft.reduce(stats, enumerate(ms_it, 1), {'loss': zero, 'speed': 0, 'samples': []})
        return {**stat, 'loss': bwd(stat['loss']), 'samples': st.median(stat['samples'])}


def make_epoch_stats():
    ravg_loss = util.make_running_avg()
    def epoch_stats(acc, batch_stats):
        loss = ravg_loss(batch_stats['loss'])
        return {'avg_loss': loss, 'batch_stats': [*acc['batch_stats'], batch_stats]}
    return epoch_stats


def train_epoch(model, params, logfile, im_it, renders_it):
    renders_it = tuple(renders_it)
    batch_it   = tuple(enumerate(util.partition(params['batch_size'], im_it), 1))
    train      = ft.partial(train_minibatch, model, params, logfile, renders_it, len(batch_it))
    return ft.reduce(make_epoch_stats(), it.starmap(train, batch_it), {'avg_loss': 0, 'batch_stats': []})


def evaluate_epoch(model, train_params, fn_trans, logfile, meta, im_it):
    im_it = tuple(im_it)
    with torch.no_grad():
        model['net'].eval()
        n_im_tp  = n_im_triplets(meta, im_it)
        fwd      = util.memoize(ft.partial(forward, model['net'], fn_trans))
        im_tp_it = util.flatten(iter_triplets(meta, im_it))
        loss_it  = it.starmap(ft.partial(triplet_loss, train_params['margin'], fwd), im_tp_it)
        return ft.reduce(make_track_stats(logfile, 'Eval', im_it, n_im_tp), loss_it, {'loss': 0, 'speed': 0})


def iter_training(model, params, logfile, images, renders_it):
    input_len = len(images['train'])
    pluck     = ft.partial(util.pluck, ['samples', 'loss', 'speed'])
    def train_and_evaluate_epoch(epoch):
        im_it = random.sample(images['train'], k=input_len)
        #^ Shuffle dataset so generated batches are different every time
        tloss, tspeed, tsamples = pluck(train_epoch(model, params, logfile, im_it, renders_it))
        fn_on_epoch()
        return {'epoch': epoch, 'tloss': tloss}
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
    util.log(f'Epoch {epoch} ended, tloss: {tloss:.2f}, vloss: {vloss:.2f}, '
           + f'tspeed {tspeed:.2f}, vspeed {vspeed:.2f} im/s\n\n', start='\n', file=logfile)
    model['save_net'](epoch)
    return {**{'epoch': epoch, 'tloss': tloss, 'vloss': vloss}, **rest}


def train(model, train_params, logfile, meta, images):
    is_learning = make_is_learning(train_params['stopping_patience'], min_delta=0.01)
    training_it = iter_training(model, train_params, logfile, meta, images)
    learning_it = it.takewhile(is_learning, util.take(train_params['max_epochs'], training_it))
    return min(map(lambda d: train_stats(model, logfile, **d), learning_it), key=ft.partial(util.pluck, ['vloss']))


def im_score(fn_fwd, meta, im, im_pairs_it):
    def pair_score(im, seg):
        distance = float(torch.cdist(fn_fwd(im), fn_fwd(seg)))
        positive = seg in meta[im]['positive']
        return {'segment': seg ,'is_pos': positive, 'dist': distance}
    return (im, sorted(it.starmap(pair_score, im_pairs_it), key=ft.partial(util.pluck, ['dist'])))


def test(model, train_params, meta, im_it):
    im_it = tuple(im_it)
    with torch.no_grad():
        model['net'].eval()
        transform = make_im_transform(model['device'], train_params['input_size'])
        fwd       = util.memoize(ft.partial(forward, model['net'], transform))
        yield from it.starmap(ft.partial(im_score, fwd, meta), zip(im_it, iter_im_pairs(meta, im_it)))

