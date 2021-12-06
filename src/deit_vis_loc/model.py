#!/usr/bin/env python3

import PIL.Image
import torch
import torchvision.transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import collections as cl
import functools   as ft
import itertools   as it
import json
import os
import random
import time
from datetime import datetime

import src.deit_vis_loc.utils as utils


def gen_anchor_imgs(dataset_dpath, name):
    queries_dpath = os.path.join(dataset_dpath, 'query_original_result')
    dataset_fpath = os.path.join(queries_dpath, name)
    return (os.path.join(queries_dpath, l.strip()) for l in open(dataset_fpath))


def gen_triplets(list_of_anchor_imgs, fn_to_segment_img):
    set_of_anchors = set(list_of_anchor_imgs)
    for anchor_img in list_of_anchor_imgs:
        positive_segment = fn_to_segment_img(anchor_img)
        for negative_img in set_of_anchors - set([anchor_img]):
            yield { 'anchor'  : anchor_img,
                    'positive': positive_segment,
                    'negative': fn_to_segment_img(negative_img) }


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
    def batch_all_triplet_loss(list_of_triplets):
        gen_losses = (triplet_loss(t) for t in list_of_triplets)
        return (l for l in gen_losses if torch.is_nonzero(l))
    return batch_all_triplet_loss


def train_epoch(model, optimizer, fn_embeddings, params, list_of_imgs):
    torch.set_grad_enabled(True)
    model.train()
    gen_loss = make_batch_all_triplet_loss(fn_embeddings, params['triplet_margin'])
    for batch in utils.partition(params['batch_size'], list_of_imgs):
        for loss in gen_loss(gen_triplets(batch, utils.to_segment_img)):
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            yield loss


def evaluate_epoch(model, fn_embeddings, params, list_of_imgs):
    torch.set_grad_enabled(False)
    model.eval();
    memoize  = ft.lru_cache(maxsize=None)
    gen_loss = make_batch_all_triplet_loss(memoize(fn_embeddings), params['triplet_margin'])
    #^ Saves re-computation of repeated images in triplets
    return gen_loss(gen_triplets(list_of_imgs, utils.to_segment_img))


def make_embeddings(model, device):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=3),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return lambda img: model(to_tensor(PIL.Image.open(img)).unsqueeze(0).to(device))


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


def train(dataset_dpath, save_dpath, params):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model  = torch.hub.load('facebookresearch/deit:main', params['deit_model'], pretrained=True)
    model.to(device)

    optimizer  = torch.optim.Adam(model.parameters(), params['learning_rate'])
    embeddings = make_embeddings(model, device)
    save_model = make_save_model(save_dpath, params)
    is_trained = make_early_stoping(params['stopping_patience'], min_delta=0.01)

    list_of_train_imgs = list(gen_anchor_imgs(dataset_dpath, 'train.txt'))
    list_of_val_imgs   = list(gen_anchor_imgs(dataset_dpath, 'val.txt'))
    utils.log('Started training with {}'.format(json.dumps(params)))

    def sum_loss(gen_loss):
        return torch.sum(torch.stack(list(gen_loss)))

    def train_loss(epoch):
        random.shuffle(list_of_train_imgs)
        #^ Shuffle dataset so generated batches are different every time
        loss = sum_loss(train_epoch(model, optimizer, embeddings, params, list_of_train_imgs))
        utils.log('Training loss for epoch {} is {}'.format(epoch, loss))
        return loss

    def val_loss(epoch):
        loss = sum_loss(evaluate_epoch(model, embeddings, params, list_of_val_imgs))
        utils.log('Validation loss for epoch {} is {}'.format(epoch, loss))
        return loss

    gen_epoch      = (cl.ChainMap({'epoch': e + 1}) for e in range(params['max_epochs']))
    gen_train_loss = (cl.ChainMap({'train_loss': train_loss(e['epoch'])}) for e in gen_epoch)
    gen_epoch_data = (cl.ChainMap({'val_loss'  : val_loss(e['epoch'])})   for e in gen_train_loss)

    for epoch_data in gen_epoch_data:
        save_model(model, epoch_data['epoch'])
        if is_trained(epoch_data['val_loss']): break

    utils.log('Finished training')


def test_model(model, fn_embeddings, list_of_anchor_imgs):
    torch.set_grad_enabled(False)
    model.eval();
    list_of_segment_imgs = [utils.to_segment_img(a) for a in list_of_anchor_imgs]
    for anchor_img in list_of_anchor_imgs:
        a_embed  = fn_embeddings(anchor_img)
        s_dists  = [torch.cdist(a_embed, fn_embeddings(s)) for s in list_of_segment_imgs]
        segments = ({'path': p, 'distance': d} for p, d in zip(list_of_segment_imgs, s_dists))
        yield { 'anchor': anchor_img, 'segments': segments }


def test(dataset_dpath, model_fpath):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model  = torch.load(model_fpath)
    model.to(device)

    memoize      = ft.lru_cache(maxsize=None)
    list_of_imgs = list(gen_anchor_imgs(dataset_dpath, 'test.txt'))
    #^ Saves re-computation of repeated images in triplets
    return test_model(model, memoize(make_embeddings(model, device)), list_of_imgs)

