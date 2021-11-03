#!/usr/bin/env python3

import PIL.Image
import torch
import torchvision.transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

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
            yield {
                'anchor'  : anchor_img,
                'positive': positive_segment,
                'negative': fn_to_segment_img(negative_img)
            }


def gen_triplet_loss(fn_triplet_loss, fn_img_to_tensor, list_of_imgs):
    list_of_triplets = list(gen_triplets(list_of_imgs, utils.to_segment_img))
    random.shuffle(list_of_triplets)
    #^ Shuffle triplets so they are not sorted by an anchor
    to_tensor_triplet = lambda t: {k: fn_img_to_tensor(v) for k,v in t.items()}
    return (fn_triplet_loss(tt) for tt in (to_tensor_triplet(t) for t in list_of_triplets))


def train_epoch(model, optimizer, fn_triplet_loss, fn_img_to_tensor, batch_size, list_of_imgs):
    model.train()

    for batch_of_imgs in utils.partition(batch_size, list_of_imgs):
        for loss in gen_triplet_loss(fn_triplet_loss, fn_img_to_tensor, batch_of_imgs):
            optimizer.zero_grad(); loss.backward(); optimizer.step()


def make_triplet_loss(model, device, margin):
    def triplet_loss(triplet):
        a_embed = model(triplet['anchor'])
        a_p_dis = torch.cdist(a_embed, model(triplet['positive']))
        a_n_dis = torch.cdist(a_embed, model(triplet['negative']))
        return torch.max(torch.tensor(0, device=device), a_p_dis - a_n_dis + margin)
    return triplet_loss


def make_img_to_tensor(device):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=3),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return lambda img: to_tensor(PIL.Image.open(img)).unsqueeze(0).to(device)


def gen_evaluate_model(model, fn_img_to_tensor, list_of_anchor_imgs):
    torch.set_grad_enabled(False);
    model.eval();

    list_of_anchor_imgs  = list(list_of_anchor_imgs)
    list_of_segment_imgs = [utils.to_segment_img(a) for a in list_of_anchor_imgs]
    compute_embeddings   = lambda im_path: model(fn_img_to_tensor(im_path))
    for anchor_img in list_of_anchor_imgs:
        a_embed = compute_embeddings(anchor_img)
        s_dists = (torch.cdist(a_embed, compute_embeddings(s)) for s in list_of_segment_imgs)
        yield { 'anchor': anchor_img, 'segments': zip(list_of_segment_imgs, s_dists) }


def make_save_model(save_dpath, params):
    time_str  = datetime.fromtimestamp(time.time()).strftime("%Y%m%dT%H%M%S")
    param_str = '-'.join(str(params[k]) for k in ['deit_model', 'batch_size'])

    os.makedirs(save_dpath, exist_ok=True)
    params_filename = '-'.join([time_str, param_str]) + '.json'
    with open(os.path.join(save_dpath, params_filename), 'w') as f:
        json.dump(params, f, indent=4)

    def save_model(model, epoch):
        model_filename = '-'.join([time_str, param_str, str(epoch + 1).zfill(3)]) + '.torch'
        torch.save(model, os.path.join(save_dpath, model_filename))
    return save_model


def train(dataset_dpath, save_dpath, params):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model  = torch.hub.load('facebookresearch/deit:main', params['deit_model'], pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])

    list_of_train_imgs = list(gen_anchor_imgs(dataset_dpath, 'train.txt'))
    random.shuffle(list_of_train_imgs)
    #^ Shuffle dataset so generated batches are different every time

    img_to_tensor = make_img_to_tensor(device)
    triplet_loss  = make_triplet_loss(model, device, margin=params['triplet_margin'])
    save_model    = make_save_model(save_dpath, params)
    for epoch in range(params['epochs']):
        train_epoch(model, optimizer, triplet_loss, img_to_tensor, params['batch_size'], list_of_train_imgs)
        save_model(model, epoch)

