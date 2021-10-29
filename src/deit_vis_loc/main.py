#!/usr/bin/env python3

import PIL.Image
import torch
import torchvision.transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
import random

import src.deit_vis_loc.utils as utils


DATASET_PATH = '.git/datasets/GeoPose3K_v2/'

def gen_query_paths(dataset_path, name):
    queries_dpath = os.path.join(dataset_path, 'query_original_result')
    dataset_fpath = os.path.join(queries_dpath, name)
    return (os.path.join(queries_dpath, l.strip()) for l in open(dataset_fpath))


def embeddings(model, fn_transform, device, fpath):
    return model(fn_transform(PIL.Image.open(fpath)).unsqueeze(0).to(device))


def gen_embedding_distances(model, fn_transform, device, list_of_query_paths):
    torch.set_grad_enabled(False);
    model.eval();

    list_of_seg_paths      = [utils.to_segment_path(p) for p in list_of_query_paths]
    list_of_seg_embeddings = [embeddings(model, fn_transform, device, seg_path)
            for seg_path in list_of_seg_paths]

    for query_path in list_of_query_paths:
        query_embedding       = embeddings(model, fn_transform, device, query_path)
        list_of_seg_distances = [torch.cdist(query_embedding, e).item()
                for e in list_of_seg_embeddings]
        list_of_seg_dist_path = [{'distance': d, 'path': p}
                for p, d in zip(list_of_seg_paths, list_of_seg_distances)]
        yield { 'query_path': query_path, 'segments': list_of_seg_dist_path }


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


def gen_tensor_triplets(fn_to_tensor, list_of_triplets):
    to_tensor_triplet = lambda t: {k: fn_to_tensor(v) for k,v in t.items()}
    return (to_tensor_triplet(t) for t in list_of_triplets)


def gen_triplet_loss(fn_triplet_loss, fn_to_tensor, list_of_imgs):
    list_of_triplets = list(gen_triplets(list_of_imgs, utils.to_segment_path))
    random.shuffle(list_of_triplets)
    #^ Shuffle triplets so they are not sorted by an anchor
    return (fn_triplet_loss(t) for t in gen_tensor_triplets(fn_to_tensor, list_of_triplets))


def train_one_epoch(model, optimizer, fn_triplet_loss, fn_to_tensor, list_of_imgs):
    model.train()
    for batch_of_imgs in utils.partition(104, list_of_imgs):
        for loss in gen_triplet_loss(fn_triplet_loss, fn_to_tensor, batch_of_imgs):
            optimizer.zero_grad(); loss.backward(); optimizer.step()


def make_triplet_loss(device, margin):
    def triplet_loss(triplet):
        a_embed = model(triplet['anchor'])
        a_p_dis = torch.cdist(a_embed, model(triplet['positive']))
        a_n_dis = torch.cdist(a_embed, model(triplet['negative']))
        return torch.max(torch.tensor(0).to(device), a_p_dis - a_n_dis + margin)
    return triplet_loss


def make_img_to_tensor(device):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=3),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return lambda im_path: to_tensor(PIL.Image.open(im_path)).unsqueeze(0).to(device)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    list_of_imgs = list(gen_query_paths(DATASET_PATH, 'train.txt'))
    random.shuffle(list_of_imgs)
    #^ Shuffle dataset so generated batches are different every time

    fn_img_to_tensor = make_img_to_tensor(device)
    fn_triplet_loss  = make_triplet_loss(device, margin=0.2)
    train_one_epoch(model, optimizer, fn_triplet_loss, fn_img_to_tensor, list_of_imgs)

