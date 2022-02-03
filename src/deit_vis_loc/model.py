#!/usr/bin/env python3

import collections as cl
import functools   as ft
import itertools   as it
import json
import operator as op
import os
import random as ra
import time
from datetime import datetime

import torch
import torch.hub
import torch.optim
import torch.cuda
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


def triplet_loss(fn_embeddings, margin, triplet):
    a_embed = fn_embeddings(triplet['anchor'])
    a_p_dis = torch.cdist(a_embed, fn_embeddings(triplet['positive']))
    a_n_dis = torch.cdist(a_embed, fn_embeddings(triplet['negative']))
    result  = a_p_dis - a_n_dis + margin
    result[0 > result] = 0
    return result


def make_iter_triplet_loss(fn_embeddings, margin):
    loss = ft.partial(triplet_loss, fn_embeddings, margin)
    return lambda triplets_it: filter(torch.is_nonzero, map(loss, triplets_it))


def backward(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def make_forward(model, fn_transform):
    return lambda fpath: model(fn_transform(fpath))


def train_batch(model_goods, triplet_margin, queries_meta, queries_it):
    transform    = util.memoize(model_goods['transform'])
    #^ Saves re-computation of im tensors, but don't cache forwarding as the model is changing
    iter_loss    = make_iter_triplet_loss(make_forward(model_goods['model'], transform), triplet_margin)
    loss_it      = iter_loss(iter_triplets(queries_meta, queries_it))
    prop_loss_it = map(ft.partial(backward, model_goods['optimizer']), loss_it)
    return ft.reduce(op.add, prop_loss_it, torch.zeros(1, 1, device=model_goods['device']))


def train_epoch(model_goods, train_params, queries_meta, queries_it):
    torch.set_grad_enabled(True)
    model_goods['model'].train();

    batch_loss    = ft.partial(train_batch, model_goods, train_params['triplet_margin'], queries_meta)
    batch_loss_it = map(batch_loss, util.partition(train_params['batch_size'], queries_it))
    return ft.reduce(op.add, batch_loss_it, torch.zeros(1, 1, device=model_goods['device']))


def evaluate_epoch(model_goods, train_params, queries_meta, queries_it):
    torch.set_grad_enabled(False)
    model_goods['model'].eval();

    forward   = util.memoize(make_forward(model_goods['model'], model_goods['transform']))
    #^ Saves re-computation of forwarding as the model is fixed during evaluation
    iter_loss = make_iter_triplet_loss(forward, train_params['triplet_margin'])
    loss_it   = iter_loss(iter_triplets(queries_meta, queries_it))
    return ft.reduce(op.add, loss_it, torch.zeros(1, 1, device=model_goods['device']))


def iter_training(model_goods, train_params, queries_meta, query_images):
    def train_and_evaluate_epoch(_):
        queries_it = ra.sample(query_images['train'], k=len(query_images['train']))
        #^ Shuffle dataset so generated batches are different every time
        train_loss = train_epoch   (model_goods, train_params, queries_meta, queries_it)
        val_loss   = evaluate_epoch(model_goods, train_params, queries_meta, query_images['val'])
        return (train_loss.item(), val_loss.item())
    return map(train_and_evaluate_epoch, it.repeat(None))


def make_im_transform(device):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=3),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return lambda fpath: to_tensor(Image.open(fpath).convert('RGB')).unsqueeze(0).to(device)


def make_save_model(save_dpath, params):
    time_str  = datetime.fromtimestamp(time.time()).strftime('%Y%m%dT%H%M%S')
    param_str = '-'.join(str(params[k]) for k in ['deit_model', 'batch_size'])
    def save_model(model, epoch):
        os.makedirs(save_dpath, exist_ok=True)
        epoch_str      = str(epoch).zfill(3)
        model_filename = '-'.join([time_str, param_str, epoch_str]) + '.torch'
        torch.save(model, os.path.join(save_dpath, model_filename))
    return save_model


def make_early_stoping(patience, min_delta):
    q_losses = cl.deque(maxlen=patience + 1)
    le_delta = lambda l, r: l - r <= min_delta
    def early_stoping(loss):
        q_losses.append(loss)
        full_queue = patience < len(q_losses)
        rest_queue = it.islice(q_losses, 1, len(q_losses))
        min_first  = all(it.starmap(le_delta, zip(it.repeat(q_losses[0]), rest_queue)))
        return full_queue and min_first
    return early_stoping


def device_name(device):
    return 'cpu' if 'cpu' == device else torch.cuda.get_device_name(device)


def train(query_images, queries_meta, train_params, output_dpath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = torch.hub.load('facebookresearch/deit:main', train_params['deit_model'], pretrained=True)
    model.to(device)
    model_goods = {
        'model'     : model,
        'device'    : device,
        'optimizer' : torch.optim.Adam(model.parameters(), train_params['learning_rate']),
        'transform' : make_im_transform(device),
    }

    util.log('Started training on "{}" with {}'.format(device_name(device), json.dumps(train_params, indent=4)))
    save_model = make_save_model(output_dpath, train_params)
    is_trained = make_early_stoping(train_params['stopping_patience'], min_delta=0.01)
    train_it   = util.take(train_params['max_epochs'], iter_training(model_goods, train_params, queries_meta, query_images))
    for epoch, (train_loss, val_loss) in enumerate(train_it, start=1):
        save_model(model, epoch)
        util.log('Loss for epoch {} is ({:0.4f}, {:0.4f})'.format(epoch, train_loss, val_loss))
        if is_trained(val_loss):
            best = epoch - train_params['stopping_patience']
            util.log('Finished training with best model in {} epoch'.format(best))
            break


def iter_test_pairs(queries_meta, queries_it):
    query_segments = lambda q: util.flatten(queries_meta[q].values())
    segments       = set(util.flatten(map(query_segments, queries_it)))
    product        = it.product(queries_it, segments)
    return ((q, set(p)) for q, p in it.groupby(product, op.itemgetter(0)))


def test_model(model, fn_embeddings, queries_meta, queries_it):
    torch.set_grad_enabled(False)
    model.eval();
    is_positive    = lambda q, s: {'is_positive': s in queries_meta[q]['positive']}
    distance       = lambda l, r: {'distance': torch.cdist(fn_embeddings(l), fn_embeddings(r))}
    build_segment  = lambda q, s: {'name': s, **distance(q, s), **is_positive(q, s)}
    test_pairs     = iter_test_pairs(queries_meta, queries_it)
    return ({'query': q, 'segments': it.starmap(build_segment, p)} for q, p in test_pairs)


def test(queries_meta, queries_it, model_fpath):
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model   = torch.load(model_fpath)
    model.to(device)
    forward = util.memoize(make_forward(model, make_im_transform(device)))
    #^ Saves re-computation of forwarding as the model is fixed during evaluation
    return test_model(model, forward, queries_meta, queries_it)

