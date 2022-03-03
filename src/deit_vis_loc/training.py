#!/usr/bin/env python3

import collections as cl
import functools   as ft
import itertools   as it
import random

import torch

import src.deit_vis_loc.util as util


def iter_im_triplets(meta, im_it):
    query_segments = lambda q: util.flatten(meta[q].values())
    segments       = set(util.flatten(map(query_segments, im_it)))
    def iter_im_triplets(query):
        query_pos = meta[query]['positive']
        query_neg = segments - query_pos
        return it.product({query}, query_pos, query_neg)
    return map(iter_im_triplets, im_it)


def triplet_loss(margin, fn_fwd, anchor, pos, neg):
    a_embed = fn_fwd(anchor)
    a_p_dis = torch.cdist(a_embed, fn_fwd(pos))
    a_n_dis = torch.cdist(a_embed, fn_fwd(neg))
    result  = a_p_dis - a_n_dis + margin
    result[0 > result] = 0
    return result


def iter_hard_im_triplets(n_im_tp, fn_tp_loss, meta, im_it):
    def iter_hard_triplets(triplets_it):
        with torch.no_grad():
            loss_it = map(lambda t: (t, float(fn_tp_loss(*t))), triplets_it)
            hard_it = map(util.first, sorted(loss_it, key=util.second, reverse=True))
            return tuple(util.take(n_im_tp, hard_it))
    return map(iter_hard_triplets, iter_im_triplets(meta, im_it))


def iter_triplet_loss(train_params, fn_fwd, meta, im_it):
    tp_loss  = ft.partial(triplet_loss, train_params['margin'], util.memoize(fn_fwd))
    im_tp_it = iter_hard_im_triplets(train_params['im_datapoints'], tp_loss, meta, im_it)
    return it.starmap(ft.partial(triplet_loss, train_params['margin'], fn_fwd), util.flatten(im_tp_it))


def backward(optimizer, loss):
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss


def forward(net, fn_transform, fpath):
    return net(fn_transform(fpath))


def make_track_stats(logfile, stage, im_it, n_im_tp):
    im_it       = tuple(im_it)
    total_ims   = len(im_it)
    avg_ims_sec = util.make_avg_ims_sec()
    formatter   = util.make_progress_formatter(bar_width=40, total=total_ims)
    ims, tps    = (0, 0)
    print(f'{formatter(stage, ims, 0)}', end='\r', file=logfile, flush=True)
    def track_stats(acc, loss):
        nonlocal ims, tps
        tps   = tps + 1
        speed = acc['speed']
        if 0 == tps % n_im_tp:
            ims   = ims + 1
            speed = avg_ims_sec(1)
            end   = '\n' if ims == total_ims else '\r'
            print(f'\033[K{formatter(stage, ims, speed)}', end=end, file=logfile, flush=True)
        return {'loss': acc['loss'] + float(loss), 'speed': speed}
        #^ Don't accumulate autograd history, hence cast the Variable to float
    return track_stats


def train_batch(model, train_params, logfile, meta, batches_total, batch_idx, im_it):
    bwd     = ft.partial(backward, model['optimizer'])
    fwd     = ft.partial(forward, model['net'], util.memoize(model['transform']))
    loss_it = map(bwd, iter_triplet_loss(train_params, fwd, meta, im_it))
    stage   = f'Batch {util.format_fraction(batch_idx, batches_total)}'
    stats   = make_track_stats(logfile, stage, im_it, train_params['im_datapoints'])
    return ft.reduce(stats, loss_it, {'loss': 0, 'speed': 0})


def make_epoch_stats():
    running_avg = util.make_running_avg()
    def epoch_stats(acc, batch_stats):
        speed = running_avg(batch_stats['speed'])
        loss  = acc['loss'] + batch_stats['loss']
        return {'loss': loss, 'speed': speed}
    return epoch_stats


def train_epoch(model, train_params, logfile, meta, im_it):
    with torch.enable_grad():
        model['net'].train()
        batches       = tuple(util.partition(train_params['batch_size'], im_it))
        batch_loss    = ft.partial(train_batch, model, train_params, logfile, meta, len(batches))
        batch_loss_it = it.starmap(batch_loss, enumerate(batches, start=1))
        return ft.reduce(make_epoch_stats(), batch_loss_it, {'loss': 0, 'speed': 0})


def n_im_triplets(meta, im_it):
    p_n_it = util.pluck(['positive', 'negative'], util.first(meta.values()))
    return util.im_triplets(*map(len, util.prepend(im_it, p_n_it)))


def evaluate_epoch(model, train_params, logfile, meta, im_it):
    im_it = tuple(im_it)
    with torch.no_grad():
        model['net'].eval()
        n_im_tp  = n_im_triplets(meta, im_it)
        fwd      = util.memoize(ft.partial(forward, model['net'], model['transform']))
        im_tp_it = util.flatten(iter_im_triplets(meta, im_it))
        loss_it  = it.starmap(ft.partial(triplet_loss, train_params['margin'], fwd), im_tp_it)
        return ft.reduce(make_track_stats(logfile, 'Eval', im_it, n_im_tp), loss_it, {'loss': 0, 'speed': 0})


def iter_training(model, train_params, logfile, meta, images):
    input_len = len(images['train'])
    pluck     = ft.partial(util.pluck, ['loss', 'speed'])
    def train_and_evaluate_epoch(epoch):
        im_it = random.sample(images['train'], k=input_len)
        #^ Shuffle dataset so generated batches are different every time
        tloss, tspeed = pluck(train_epoch   (model, train_params, logfile, meta, im_it))
        vloss, vspeed = pluck(evaluate_epoch(model, train_params, logfile, meta, images['val']))
        return {'epoch': epoch, 'tloss': tloss, 'vloss': vloss, 'tspeed': tspeed, 'vspeed': vspeed}
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


def iter_test_pairs(meta, im_it):
    query_segments = lambda q: util.flatten(meta[q].values())
    segments       = set(util.flatten(map(query_segments, im_it)))
    product        = it.product(im_it, segments)
    return ((q, set(p)) for q, p in it.groupby(product, util.first))


def eval(model, meta, im_it):
    fwd = util.memoize(ft.partial(forward, model['net'], model['transform']))
    with torch.no_grad():
        model['net'].eval()
        is_positive    = lambda q, s: {'is_positive': s in meta[q]['positive']}
        distance       = lambda l, r: {'distance': torch.cdist(fwd(l), fwd(r))}
        build_segment  = lambda q, s: {'name': s, **distance(q, s), **is_positive(q, s)}
        test_pairs     = iter_test_pairs(meta, im_it)
        return ({'query': q, 'segments': it.starmap(build_segment, p)} for q, p in test_pairs)

