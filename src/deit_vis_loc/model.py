#!/usr/bin/env python3

import torch
import torchvision.transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import collections as cl
import functools   as ft
import itertools   as it
import json
import operator as op
import os
import random
import time
from datetime import datetime

import src.deit_vis_loc.util as util


def gen_triplets(list_of_query_imgs, rendered_segments):
    query_segments = lambda query: it.chain.from_iterable(rendered_segments[query].values())
    all_segments   = set(it.chain.from_iterable(map(query_segments, list_of_query_imgs)))
    def images_of_triplet(query):
        query_pos = rendered_segments[query]['positive']
        query_neg = all_segments - query_pos
        return ([query], query_pos, query_neg)

    gen_trip = map(images_of_triplet, list_of_query_imgs)
    gen_prod = it.chain.from_iterable(it.product(q, p, n) for q, p, n in gen_trip)
    return ({'anchor': q, 'positive': p, 'negative': n}   for q, p, n in gen_prod)


def make_triplet_loss(fn_embeddings, margin):
    def triplet_loss(triplet):
        a_embed = fn_embeddings(triplet['anchor'])
        a_p_dis = torch.cdist(a_embed, fn_embeddings(triplet['positive']))
        a_n_dis = torch.cdist(a_embed, fn_embeddings(triplet['negative']))
        result  = a_p_dis - a_n_dis + margin
        result[0 > result] = 0
        return result
    return triplet_loss


def make_batch_all_triplet_loss(fn_embeddings, margin):
    triplet_loss = make_triplet_loss(fn_embeddings, margin)
    return lambda list_triplets: filter(torch.is_nonzero, map(triplet_loss, list_triplets))


def train_epoch(model, optimizer, fn_embeddings, params, list_queries, rendered_segments):
    torch.set_grad_enabled(True)
    model.train()
    gen_loss = make_batch_all_triplet_loss(fn_embeddings, params['triplet_margin'])
    for batch in util.partition(params['batch_size'], list_queries):
        for loss in gen_loss(gen_triplets(batch, rendered_segments)):
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            yield loss


def evaluate_epoch(model, fn_embeddings, params, list_of_queries, rendered_segments):
    torch.set_grad_enabled(False)
    model.eval();

    embeddings = ft.lru_cache(maxsize=None)(fn_embeddings)
    #^ Saves re-computation of repeated images in triplets
    gen_loss   = make_batch_all_triplet_loss(embeddings, params['triplet_margin'])
    return gen_loss(gen_triplets(list_of_queries, rendered_segments))


def make_embeddings(model, device):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=3),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return lambda img: model(to_tensor(Image.open(img).convert('RGB')).unsqueeze(0).to(device))


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


def train(query_images, rendered_segments, model_params, output_dpath):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model  = torch.hub.load('facebookresearch/deit:main', model_params['deit_model'], pretrained=True)
    model.to(device)

    optimizer  = torch.optim.Adam(model.parameters(), model_params['learning_rate'])
    embeddings = make_embeddings(model, device)
    save_model = make_save_model(output_dpath, model_params)
    is_trained = make_early_stoping(model_params['stopping_patience'], min_delta=0.01)
    sum_loss   = lambda gen_loss: torch.sum(torch.stack(list(gen_loss)))

    def train_loss(epoch):
        random.shuffle(query_images['train'])
        #^ Shuffle dataset so generated batches are different every time
        loss = sum_loss(train_epoch(model,
            optimizer, embeddings, model_params, query_images['train'], rendered_segments))
        util.log('Training loss for epoch {} is {}'.format(epoch, loss))
        return loss

    def val_loss(epoch):
        loss = sum_loss(evaluate_epoch(model,
            embeddings, model_params, query_images['val'], rendered_segments))
        util.log('Validation loss for epoch {} is {}'.format(epoch, loss))
        return loss

    gen_epoch      = ({'epoch': e + 1} for e in range(model_params['max_epochs']))
    gen_train_loss = ({**e, **{'train_loss': train_loss(e['epoch'])}} for e in gen_epoch)
    gen_epoch_data = ({**e, **{'val_loss'  : val_loss(e['epoch'])}}   for e in gen_train_loss)

    util.log('Started training with {}'.format(json.dumps(model_params)))
    for epoch_data in gen_epoch_data:
        save_model(model, epoch_data['epoch'])
        if is_trained(epoch_data['val_loss']): break

    util.log('Finished training')


def gen_test_pairs(list_of_query_imgs, rendered_segments):
    query_segments        = lambda q: it.chain.from_iterable(rendered_segments[q].values())
    list_of_test_segments = list(it.chain.from_iterable(map(query_segments, list_of_query_imgs)))
    query_segment_product = it.product(list_of_query_imgs, list_of_test_segments)
    return ((q, list(p)) for q, p in it.groupby(query_segment_product, op.itemgetter(0)))


def test_model(model, fn_embeddings, list_of_query_imgs, rendered_segments):
    torch.set_grad_enabled(False)
    model.eval();
    is_positive    = lambda q, s: {'is_positive': s in rendered_segments[q]['positive']}
    distance       = lambda l, r: {'distance': torch.cdist(fn_embeddings(l), fn_embeddings(r))}
    build_segment  = lambda q, s: {'name': s, **distance(q, s), **is_positive(q, s)}
    test_pairs     = gen_test_pairs(list_of_query_imgs, rendered_segments)
    return ({'query': q, 'segments': it.starmap(build_segment, p)} for q, p in test_pairs)


def test(list_of_query_imgs, rendered_segments, model_fpath):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model  = torch.load(model_fpath)
    model.to(device)
    embeddings = ft.lru_cache(maxsize=None)(make_embeddings(model, device))
    #^ Saves re-computation of repeated images in triplets
    return test_model(model, embeddings, list_of_query_imgs, rendered_segments)

