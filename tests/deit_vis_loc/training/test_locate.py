#!/usr/bin/env python3

import functools as ft
import torch

import src.deit_vis_loc.training.locate as locate


def test_iter_triplets():
    im_pos   = lambda im, render: render == f'{im}_p'
    im_neg   = lambda im, render: render != f'{im}_p'
    pos_fn   = lambda im, renders_it: filter(ft.partial(im_pos, im), renders_it)
    neg_fn   = lambda im, renders_it: filter(ft.partial(im_neg, im), renders_it)
    triplets = ft.partial(locate.iter_triplets, pos_fn, neg_fn)

    assert list(map(set, triplets([],     [])))       == []
    assert list(map(set, triplets(['i1'], [])))       == [set()]
    assert list(map(set, triplets([],     ['i1_p']))) == []

    im_it = ['i1']
    rd_it = ['i1_p', 'i1_n']
    assert list(map(set, triplets(im_it, rd_it))) == [
        {('i1', 'i1_p', 'i1_n')}
    ]
    im_it = ['i1', 'i2']
    rd_it = ['i1_p', 'i1_n', 'i2_p', 'i2_n']
    assert list(map(set, triplets(im_it, rd_it))) == [
        {('i1', 'i1_p', 'i1_n'), ('i1', 'i1_p', 'i2_p'), ('i1', 'i1_p', 'i2_n')},
        {('i2', 'i2_p', 'i2_n'), ('i2', 'i2_p', 'i1_n'), ('i2', 'i2_p', 'i1_p')},
    ]


def test_cosine_dist():
    o = torch.ones ((1, 2))
    z = torch.zeros((1, 2))
    assert locate.cosine_dist(o, o) == 0
    assert locate.cosine_dist(z, z) == 1
    assert locate.cosine_dist(o, z) == 1
    assert locate.cosine_dist(z, o) == 1


def test_triplet_loss():
    p  = {'path': torch.ones ((1, 2))}
    n  = {'path': torch.zeros((1, 2))}

    loss = ft.partial(locate.triplet_loss, 0., lambda x: x)
    assert loss(p, p, p) == torch.tensor([0.])
    assert loss(p, p, n) == torch.tensor([0.])
    assert loss(p, n, p) == torch.tensor([1.])
    assert loss(p, n, n) == torch.tensor([0.])
    assert loss(n, n, n) == torch.tensor([0.])

    loss = ft.partial(locate.triplet_loss, .5, lambda x: x)
    assert loss(p, p, p) == torch.tensor([.5])
    assert loss(p, p, n) == torch.tensor([0.])
    assert loss(p, n, n) == torch.tensor([.5])
    assert loss(n, n, n) == torch.tensor([.5])


def test_early_stopping():
    it_learning = locate.make_is_learning(0, 0)
    assert it_learning({'val': {'loss': 0}}) == False
    #^ Not having patience means to stop immediately

    it_learning = locate.make_is_learning(1, .1)
    assert it_learning({'val': {'loss': 3.00}}) == True
    assert it_learning({'val': {'loss': 2.91}}) == False
    #^ Validation loss doesn't decrease more than delta

    it_learning = locate.make_is_learning(2, 0)
    assert it_learning({'val': {'loss': 3}}) == True
    assert it_learning({'val': {'loss': 2}}) == True
    assert it_learning({'val': {'loss': 1}}) == True
    assert it_learning({'val': {'loss': 2}}) == True
    assert it_learning({'val': {'loss': 0}}) == True
    assert it_learning({'val': {'loss': 1}}) == True
    assert it_learning({'val': {'loss': 2}}) == False
    #^ Patience over multiple losses

