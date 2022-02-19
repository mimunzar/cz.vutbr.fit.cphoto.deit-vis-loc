#!/usr/bin/env python3

import collections as cl
import functools   as ft
import itertools   as it
import random

import torch

import src.deit_vis_loc.util as util


def make_triplets(queries_meta, queries_it):
    query_segments = lambda q: util.flatten(queries_meta[q].values())
    segments       = set(util.flatten(map(query_segments, queries_it)))
    def triplets(query):
        query_pos = queries_meta[query]['positive']
        query_neg = segments - query_pos
        return it.product({query}, query_pos, query_neg)
    return triplets


def iter_triplets(queries_meta, queries_it):
    triplets = make_triplets(queries_meta, queries_it)
    labels   = lambda q, p, n: {'anchor': q, 'positive': p, 'negative': n}
    return it.starmap(labels, util.flatten(map(triplets, queries_it)))


def triplet_loss(margin, fn_forward, triplet):
    a_embed = fn_forward(triplet['anchor'])
    a_p_dis = torch.cdist(a_embed, fn_forward(triplet['positive']))
    a_n_dis = torch.cdist(a_embed, fn_forward(triplet['negative']))
    result  = a_p_dis - a_n_dis + margin
    result[0 > result] = 0
    return result


def make_iter_triplet_loss(margin, fn_forward, fn_iter_triplets=iter_triplets):
    loss = ft.partial(triplet_loss, margin, fn_forward)
    def iter_triplet_loss(queries_meta, queries_it):
        return map(loss, fn_iter_triplets(queries_meta, queries_it))
    return iter_triplet_loss


def backward(optimizer, loss):
    if (torch.is_nonzero(loss)):
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss


def forward(net, fn_transform, fpath):
    return net(fn_transform(fpath))


def im_triplets(queries_meta, queries_it):
    seg_it = util.pluck(['positive', 'negative'], util.first(queries_meta.values()))
    return util.im_triplets(*map(len, util.prepend(queries_it, seg_it)))


def make_track_stats(logfile, stage, queries_meta, queries_it):
    queries_it  = tuple(queries_it)
    total_ims   = len(queries_it)
    im_tps      = im_triplets(queries_meta, queries_it)
    avg_ims_sec = util.make_avg_ims_sec()
    formatter   = util.make_progress_formatter(bar_width=40, total=total_ims)
    ims, tps    = (0, 0)
    print(f'{formatter(stage, ims, 0)}', end='\r', file=logfile)
    def track_stats(acc, loss):
        nonlocal ims, tps
        tps   = tps + 1
        speed = acc['speed']
        if 0 == tps % im_tps:
            ims   = ims + 1
            speed = avg_ims_sec(1)
            print(f'\033[K{formatter(stage, ims, speed)}', end='\n' if ims == total_ims else '\r', file=logfile)
        return {'loss': acc['loss'] + float(loss), 'speed': speed}
        #^ Don't accumulate autograd history, hence cast the Variable to float
    return track_stats


def train_batch(model, train_params, logfile, queries_meta, batches_total, batch_idx, queries_it):
    stage_str = f'Batch {util.format_fraction(batch_idx, batches_total)}'
    transform = util.memoize(model['transform'])
    #^ Save re-computation of image tensors, but don't cache forwarding as network is changing
    iter_loss = make_iter_triplet_loss(train_params['triplet_margin'], ft.partial(forward, model['net'], transform))
    loss_it   = map(ft.partial(backward, model['optimizer']), iter_loss(queries_meta, queries_it))
    return ft.reduce(make_track_stats(logfile, stage_str, queries_meta, queries_it), loss_it, {'loss': 0, 'speed': 0})


def make_epoch_stats():
    running_avg = util.make_running_avg()
    def epoch_stats(acc, batch_stats):
        speed = running_avg(batch_stats['speed'])
        loss  = acc['loss'] + batch_stats['loss']
        return {'loss': loss, 'speed': speed}
    return epoch_stats


def train_epoch(model, train_params, logfile, queries_meta, queries_it):
    queries_it = tuple(queries_it)
    with torch.enable_grad():
        model['net'].train()
        batches       = tuple(util.partition(train_params['batch_size'], queries_it))
        batch_loss    = ft.partial(train_batch, model, train_params, logfile, queries_meta, len(batches))
        batch_loss_it = it.starmap(batch_loss, enumerate(batches, start=1))
        return ft.reduce(make_epoch_stats(), batch_loss_it, {'loss': 0, 'speed': 0})


def evaluate_epoch(model, train_params, logfile, queries_meta, queries_it):
    queries_it = tuple(queries_it)
    with torch.no_grad():
        model['net'].eval()
        stats     = make_track_stats(logfile, 'Eval', queries_meta, queries_it)
        fwd       = util.memoize(ft.partial(forward, model['net'], model['transform']))
        #^ Saves re-computation of forwarding as network is fixed during evaluation
        iter_loss = make_iter_triplet_loss(train_params['triplet_margin'], fwd)
        return ft.reduce(stats, iter_loss(queries_meta, queries_it), {'loss': 0, 'speed': 0})


def iter_training(model, train_params, logfile, queries_meta, query_images):
    input_len = len(query_images['train'])
    pluck     = ft.partial(util.pluck, ['loss', 'speed'])
    def train_and_evaluate_epoch(epoch):
        queries_it = random.sample(query_images['train'], k=input_len)
        #^ Shuffle dataset so generated batches are different every time
        tloss, tspeed = pluck(train_epoch   (model, train_params, logfile, queries_meta, queries_it))
        vloss, vspeed = pluck(evaluate_epoch(model, train_params, logfile, queries_meta, query_images['val']))
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


def train(model, train_params, logfile, queries_meta, query_images):
    is_learning = make_is_learning(train_params['stopping_patience'], min_delta=0.01)
    training_it = iter_training(model, train_params, logfile, queries_meta, query_images)
    learning_it = it.takewhile(is_learning, util.take(train_params['max_epochs'], training_it))
    return min(map(lambda d: train_stats(model, logfile, **d), learning_it), key=ft.partial(util.pluck, ['vloss']))


def iter_test_pairs(queries_meta, queries_it):
    query_segments = lambda q: util.flatten(queries_meta[q].values())
    segments       = set(util.flatten(map(query_segments, queries_it)))
    product        = it.product(queries_it, segments)
    return ((q, set(p)) for q, p in it.groupby(product, util.first))


def eval(model, queries_meta, queries_it):
    fwd = util.memoize(ft.partial(forward, model['net'], model['transform']))
    #^ Saves re-computation of forwarding as network is fixed during evaluation
    with torch.no_grad():
        model['net'].eval()
        is_positive    = lambda q, s: {'is_positive': s in queries_meta[q]['positive']}
        distance       = lambda l, r: {'distance': torch.cdist(fwd(l), fwd(r))}
        build_segment  = lambda q, s: {'name': s, **distance(q, s), **is_positive(q, s)}
        test_pairs     = iter_test_pairs(queries_meta, queries_it)
        return ({'query': q, 'segments': it.starmap(build_segment, p)} for q, p in test_pairs)

