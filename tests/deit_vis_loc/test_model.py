#!/usr/bin/env python3

import src.deit_vis_loc.model as model

import torch
from operator import itemgetter


def test_generate_triplets():
    fn_to_segment_path = lambda s: s + '_segment'
    assert list(model.gen_triplets([], fn_to_segment_path)) == []
    assert list(model.gen_triplets(['foo'], fn_to_segment_path)) == []
    #^ No negative samples present
    triplets = model.gen_triplets(['foo', 'bar', 'baz'], fn_to_segment_path)
    assert sorted(triplets, key=itemgetter('anchor', 'negative')) == [
        {'anchor': 'bar', 'positive': 'bar_segment', 'negative': 'baz_segment'},
        {'anchor': 'bar', 'positive': 'bar_segment', 'negative': 'foo_segment'},
        {'anchor': 'baz', 'positive': 'baz_segment', 'negative': 'bar_segment'},
        {'anchor': 'baz', 'positive': 'baz_segment', 'negative': 'foo_segment'},
        {'anchor': 'foo', 'positive': 'foo_segment', 'negative': 'bar_segment'},
        {'anchor': 'foo', 'positive': 'foo_segment', 'negative': 'baz_segment'},
    ]


def test_triplet_loss():
    l = model.make_triplet_loss(lambda x: x, margin=0)
    z = torch.zeros(1, 1)
    o = torch.ones (1, 1)
    assert l({'anchor': z, 'positive': z, 'negative': z}) == torch.tensor([[0.]])
    assert l({'anchor': z, 'positive': z, 'negative': o}) == torch.tensor([[0.]])
    assert l({'anchor': z, 'positive': o, 'negative': z}) == torch.tensor([[1.]])
    assert l({'anchor': z, 'positive': o, 'negative': o}) == torch.tensor([[0.]])
    assert l({'anchor': o, 'positive': o, 'negative': o}) == torch.tensor([[0.]])

    l = model.make_triplet_loss(lambda x: x, margin=0.5)
    assert l({'anchor': z, 'positive': z, 'negative': z}) == torch.tensor([[0.5]])
    assert l({'anchor': z, 'positive': z, 'negative': o}) == torch.tensor([[0.]])
    assert l({'anchor': z, 'positive': o, 'negative': z}) == torch.tensor([[1.5]])
    assert l({'anchor': z, 'positive': o, 'negative': o}) == torch.tensor([[0.5]])
    assert l({'anchor': o, 'positive': o, 'negative': o}) == torch.tensor([[0.5]])


def test_batch_all_triplet_loss():
    gen_l = model.make_batch_all_triplet_loss(lambda x: x, margin=0.5)
    z = torch.zeros(1, 1)
    o = torch.ones (1, 1)
    assert list(gen_l([])) == []
    assert list(gen_l([
            {'anchor': z, 'positive': z, 'negative': z},
            {'anchor': z, 'positive': z, 'negative': o},
            {'anchor': z, 'positive': o, 'negative': z},
            {'anchor': z, 'positive': o, 'negative': o},
            {'anchor': o, 'positive': o, 'negative': o},
        ])) == [torch.tensor([[0.5]]),
                torch.tensor([[1.5]]),
                torch.tensor([[0.5]]),
                torch.tensor([[0.5]])]
    #^ Same as in triplet_loss test but filters out zero losses

