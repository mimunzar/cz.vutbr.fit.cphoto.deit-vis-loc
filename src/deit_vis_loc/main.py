#!/usr/bin/env python3

import PIL.Image
import torch
import torchvision.transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
import random

import src.deit_vis_loc.utils as utils


DATASET_PATH = '.git/datasets/GeoPose3K_v2/'
BATCH_SIZE   = 100


def gen_query_paths(dataset_path, name):
    queries_dpath = os.path.join(dataset_path, 'query_original_result')
    dataset_fpath = os.path.join(queries_dpath, name)
    if os.path.exists(dataset_fpath):
        return (os.path.join(queries_dpath, l.strip()) for l in open(dataset_fpath))
    raise FileNotFoundError(
            'Failed to read queries paths ({} not found in {})'.format(name, queries_dpath))


def embeddings(model, fn_transform, fpath):
    return model(fn_transform(PIL.Image.open(fpath)).unsqueeze(0))


def gen_embedding_distances(model, fn_transform, list_of_query_paths):
    torch.set_grad_enabled(False);
    model.eval();

    list_of_seg_paths      = [utils.to_segment_path(p) for p in list_of_query_paths]
    list_of_seg_embeddings = [embeddings(model, fn_transform, seg_path)
            for seg_path in list_of_seg_paths]

    for query_path in list_of_query_paths:
        query_embedding       = embeddings(model, fn_transform, query_path)
        list_of_seg_distances = [torch.cdist(query_embedding, e).item()
                for e in list_of_seg_embeddings]
        list_of_seg_dist_path = [{'distance': d, 'path': p}
                for p, d in zip(list_of_seg_paths, list_of_seg_distances)]
        yield { 'query_path': query_path, 'segments': list_of_seg_dist_path }


def gen_triplets(list_of_query_paths, fn_to_segment_path):
    set_of_query_paths = set(list_of_query_paths)
    for query_path in list_of_query_paths:
        pos_segment = fn_to_segment_path(query_path)
        for neg_path in set_of_query_paths - set([query_path]):
            yield {
                'anchor'  : query_path,
                'positive': pos_segment,
                'negative': fn_to_segment_path(neg_path)
            }


def gen_loss(model, fn_transform, list_of_query_paths):
    list_of_triplets = list(gen_triplets(list_of_query_paths, utils.to_segment_path))
    random.shuffle(list_of_triplets)
    for tp in list_of_triplets:
        a_embed = embeddings(model, fn_transform, tp['anchor'])
        a_p_dis = torch.cdist(a_embed, embeddings(model, fn_transform, tp['positive']))
        a_n_dis = torch.cdist(a_embed, embeddings(model, fn_transform, tp['negative']))
        yield torch.max(torch.tensor(0), a_p_dis - a_n_dis + 0.2)


def train_one_batch(model, optimizer, fn_transform, list_of_query_paths):
    model.train()
    for loss in gen_loss(model, fn_transform, list_of_query_paths):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    model     = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=3),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    batch = utils.partition(5, gen_query_paths(DATASET_PATH, 'train.txt'))
    train_one_batch(model, optimizer, transform, next(batch))

