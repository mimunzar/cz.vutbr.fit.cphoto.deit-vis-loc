#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
from operator import itemgetter


DATASET_PATH = '.git/datasets/GeoPose3K_v2/'
BATCH_SIZE   = 100


def read_queries_paths(dataset_path, name):
    queries_dpath = os.path.join(dataset_path, 'query_original_result')
    dataset_fpath = os.path.join(queries_dpath, name)
    if os.path.exists(dataset_fpath):
        with open(dataset_fpath) as f:
            return [os.path.join(queries_dpath, line.strip()) for line in f]
    raise FileNotFoundError(
            'Failed to read queries paths ({} not found in {})'.format(name, queries_dpath))


def partition(n, coll):
    assert 0 < n
    if not coll:
        return []
    part = coll[:n]
    if len(part) == n:
        return [part] + partition(n, coll[n:])
    return [part]


def to_segment_path(query_path):
    return query_path \
        .replace('query_original_result', 'query_segments_result') \
        .replace('.jpg', '.png')


def gen_triplets(list_of_query_paths, fn_to_segment_path):
    result             = []
    set_of_query_paths = set(list_of_query_paths)
    for query_path in list_of_query_paths:
        pos_segment = fn_to_segment_path(query_path)
        for neg_path in set_of_query_paths - set([query_path]):
            result.append({
                'A': query_path,
                'P': pos_segment,
                'N': fn_to_segment_path(neg_path)
            })
    return result


def embeddings(model, fn_transform, fpath):
    return model(fn_transform(Image.open(fpath)).unsqueeze(0))


def gen_embedding_distances(model, fn_transform, list_of_query_paths):
    torch.set_grad_enabled(False);
    model.eval();

    list_of_seg_paths      = [to_segment_path(p) for p in list_of_query_paths]
    list_of_seg_embeddings = [embeddings(model, fn_transform, seg_path)
            for seg_path in list_of_seg_paths]

    for query_path in list_of_query_paths:
        query_embedding       = embeddings(model, fn_transform, query_path)
        list_of_seg_distances = [torch.cdist(query_embedding, e).item()
                for e in list_of_seg_embeddings]
        list_of_seg_dist_path = [{'distance': d, 'path': p}
                for p, d in zip(list_of_seg_paths, list_of_seg_distances)]
        yield { 'query_path': query_path, 'segments': list_of_seg_dist_path }


def plot_image(axis, fpath, border=None):
    if border:
        for spine_pos in ['bottom','top', 'right', 'left']:
            axis.spines[spine_pos].set_color(border['color'])
            axis.spines[spine_pos].set_linewidth(border['width'])
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.imshow(mpimg.imread(fpath))


def plot_closest_distances(an_embedding_distance):
    query_path = an_embedding_distance['query_path']
    fig, axis  = plt.subplots(1, 5, constrained_layout=True)
    plot_image(axis[0], query_path)

    list_of_segments = sorted(
            an_embedding_distance['segments'], key=itemgetter('distance'))
    for i in range(1, 5):
        segment_path = list_of_segments[i - 1]['path']
        border_color = 'green' if segment_path == to_segment_path(query_path) else 'red'
        plot_image(axis[i], segment_path, border={'color': border_color, 'width': 3})


if __name__ == "__main__":
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

    transform = T.Compose([
        T.Resize(256, interpolation=3),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    result = gen_embedding_distances(model, transform, read_queries_paths(DATASET_PATH, 'val.txt'))

