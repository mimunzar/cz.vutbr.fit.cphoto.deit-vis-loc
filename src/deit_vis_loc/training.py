#!/usr/bin/env python3

import collections as cl
import functools   as ft
import itertools   as it
import operator    as op
import os
import random      as ra
from datetime import datetime
import tracemalloc

import torch
import torchvision.transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import src.deit_vis_loc.util as util


def make_qpn(queries_meta, queries_it):
    query_segments = lambda q: util.flatten(queries_meta[q].values())
    segments       = set(util.flatten(map(query_segments, queries_it)))
    def qpn(query):
        query_pos = queries_meta[query]['positive']
        query_neg = segments - query_pos
        return ({query}, query_pos, query_neg)
    return qpn


def iter_triplets(queries_meta, queries_it):
    label  = lambda q, p, n: {'anchor': q, 'positive': p, 'negative': n}
    qpn_it = map(make_qpn(queries_meta, queries_it), queries_it)
    trp_it = util.flatten(it.starmap(it.product, qpn_it))
    return it.starmap(label, trp_it)


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
        triplets_it = fn_iter_triplets(queries_meta, queries_it)
        pos_loss_it = filter(torch.is_nonzero, map(loss, triplets_it))
        return pos_loss_it
    return iter_triplet_loss


def backward(optimizer, loss):
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss


def forward(model, fn_transform, fpath):
    return model(fn_transform(fpath))


def train_batch(model_goods, margin, queries_meta, queries_it):
    transform = util.memoize(model_goods['transform'])
    #^ Saves re-computation of image tensors, but don't cache forwarding as the model is changing
    iter_loss = make_iter_triplet_loss(margin, ft.partial(forward, model_goods['model'], transform))
    loss_it   = map(ft.partial(backward, model_goods['optimizer']), iter_loss(queries_meta, queries_it))
    return ft.reduce(op.add, loss_it, torch.zeros(1, 1, device=model_goods['device']))


def make_batch_stats(batch_size, queries_it):
    progress_bar = ft.partial(util.progress_bar, 40, len(tuple(queries_it)))
    running_avg  = util.make_running_avg()
    ims_sec      = util.make_ims_sec()
    batch_idx    = 0
    print(f'\033[K  {progress_bar(0)}  (0.00 im/sec)', end='\r')
    def batch_stats(acc, loss):
        nonlocal batch_idx
        batch_idx   = batch_idx + 1
        done_ims    = batch_idx*batch_size
        avg_ims_sec = running_avg(ims_sec(done_ims))
        print(f'\033[K  {progress_bar(done_ims)}  ({avg_ims_sec:.02} im/sec)', end='\r')
        return {'loss': acc['loss'] + loss, 'speed': avg_ims_sec}
    return batch_stats


def train_epoch(model_goods, train_params, queries_meta, queries_it):
    queries_it    = tuple(queries_it)
    model, device = util.pluck(['model', 'device'], model_goods)
    with torch.enable_grad():
        model.train()
        batch_stats   = make_batch_stats(train_params['batch_size'], queries_it)
        batch_loss    = ft.partial(train_batch, model_goods, train_params['triplet_margin'], queries_meta)
        batch_loss_it = map(batch_loss, util.partition(train_params['batch_size'], queries_it))
        return ft.reduce(batch_stats, batch_loss_it, {'loss': torch.zeros(1, 1, device=device), 'speed': 0})


def evaluate_epoch(model_goods, train_params, queries_meta, queries_it):
    queries_it    = tuple(queries_it)
    model, device = util.pluck(['model', 'device'], model_goods)
    margin        = util.pluck(['triplet_margin'], train_params)
    with torch.no_grad():
        model.eval()
        c_forward   = util.memoize(ft.partial(forward, model, model_goods['transform']))
        #^ Saves re-computation of forwarding as the model is fixed during evaluation
        val_loss_it = make_iter_triplet_loss(margin, c_forward)(queries_meta, queries_it)
        return ft.reduce(op.add, val_loss_it, torch.zeros(1, 1, device=device))


def iter_training(model_goods, train_params, queries_meta, query_images):
    input_len = len(query_images['train'])
    pluck     = ft.partial(util.pluck, ['loss', 'speed'])
    def train_and_evaluate_epoch(epoch):
        queries_it = ra.sample(query_images['train'], k=input_len)
        #^ Shuffle dataset so generated batches are different every time
        tloss, tspeed = pluck(train_epoch(model_goods, train_params, queries_meta, queries_it))
        vloss         = evaluate_epoch(model_goods, train_params, queries_meta, query_images['val'])
        return {'epoch': epoch, 'train': tloss.item(), 'val': vloss.item(), 'speed': tspeed}
    return map(train_and_evaluate_epoch, it.count(1))


def make_is_learning(patience, min_delta):
    q_losses = cl.deque(maxlen=patience + 1)
    le_delta = lambda l, r: l - r <= min_delta
    def is_training(losses):
        q_losses.append(losses['val'])
        full_queue = patience < len(q_losses)
        rest_queue = it.islice(q_losses, 1, len(q_losses))
        min_first  = all(it.starmap(le_delta, zip(it.repeat(q_losses[0]), rest_queue)))
        return not (full_queue and min_first)
    return is_training


def make_save_model(save_dpath, params):
    time_str  = datetime.fromtimestamp(util.epoch_secs()).strftime('%Y%m%dT%H%M%S')
    param_str = '-'.join(str(params[k]) for k in ['deit_model', 'batch_size'])
    def save_model(model, epoch):
        os.makedirs(save_dpath, exist_ok=True)
        epoch_str      = str(epoch).zfill(3)
        model_filename = '-'.join([time_str, param_str, epoch_str]) + '.torch'
        torch.save(model, os.path.join(save_dpath, model_filename))
    return save_model


def make_epoch_IO(model_goods, train_params, output_dpath):
    save_model = make_save_model(output_dpath, train_params)
    def epoch_IO(epoch, train, val, speed, **rest):
        util.log(f'Epoch {epoch} ended, tloss: {train:.4f}, vloss: {val:.4f}, speed {speed:.2f} im/sec')
        mem_size, peak_size = map(lambda x: x/1024, tracemalloc.get_traced_memory())
        util.log(f'Mem size {mem_size:.1f} KiB, peak size {peak_size:.1f} KiB')
        tracemalloc.take_snapshot().dump(f'snapshot_{epoch}.mem')
        save_model(model_goods['model'], epoch)
        return {**{'epoch': epoch, 'train': train, 'val': val}, **rest}
    return epoch_IO


def train(model_goods, train_params, queries_meta, query_images, output_dpath):
    epoch_IO    = make_epoch_IO(model_goods, train_params, output_dpath)
    is_learning = make_is_learning(train_params['stopping_patience'], min_delta=0.01)
    training_it = iter_training(model_goods, train_params, queries_meta, query_images)
    learning_it = it.takewhile(is_learning, util.take(train_params['max_epochs'], training_it))
    return min(map(lambda d: epoch_IO(**d), learning_it), key=lambda losses: losses['val'])


def iter_test_pairs(queries_meta, queries_it):
    query_segments = lambda q: util.flatten(queries_meta[q].values())
    segments       = set(util.flatten(map(query_segments, queries_it)))
    product        = it.product(queries_it, segments)
    return ((q, set(p)) for q, p in it.groupby(product, op.itemgetter(0)))


def eval_model(model_goods, fn_forward, queries_meta, queries_it):
    with torch.no_grad():
        model_goods['model'].eval()
        is_positive    = lambda q, s: {'is_positive': s in queries_meta[q]['positive']}
        distance       = lambda l, r: {'distance': torch.cdist(fn_forward(l), fn_forward(r))}
        build_segment  = lambda q, s: {'name': s, **distance(q, s), **is_positive(q, s)}
        test_pairs     = iter_test_pairs(queries_meta, queries_it)
        return ({'query': q, 'segments': it.starmap(build_segment, p)} for q, p in test_pairs)


def make_im_transform(device, input_size=224):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size, interpolation=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return lambda fpath: to_tensor(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)


def eval(model_goods, queries_meta, queries_it):
    transform = make_im_transform(model_goods['device'])
    m_forward = util.memoize(ft.partial(forward, model_goods['model'], transform))
    #^ Saves re-computation of forwarding as the model is fixed during evaluation
    return eval_model(model_goods, m_forward, queries_meta, queries_it)

