#!/usr/bin/env python3

import functools as ft
import torch

import src.deit_vis_loc.training as training



def test_iter_triplets():
    im_pos   = lambda im, render: render == f'{im}_p'
    im_neg   = lambda im, render: render != f'{im}_p'
    pos_fn   = lambda im, renders_it: filter(ft.partial(im_pos, im), renders_it)
    neg_fn   = lambda im, renders_it: filter(ft.partial(im_neg, im), renders_it)
    triplets = ft.partial(training.iter_triplets, pos_fn, neg_fn)

    assert list(map(set, triplets({}, []))) == []
    im_it = ['foo']
    rd_it = ['foo_p', 'foo_n']
    assert list(map(set, triplets(im_it, rd_it))) == [
            {('foo', 'foo_p', 'foo_n')}
        ]
    im_it = ['foo', 'bar']
    rd_it = ['foo_p', 'foo_n', 'bar_p', 'bar_n']
    assert list(map(set, triplets(im_it, rd_it))) == [
            {('foo', 'foo_p', 'foo_n'), ('foo', 'foo_p', 'bar_p'), ('foo', 'foo_p', 'bar_n')},
            {('bar', 'bar_p', 'bar_n'), ('bar', 'bar_p', 'foo_n'), ('bar', 'bar_p', 'foo_p')},
        ]


def test_triplet_loss():
    p = {'path': torch.tensor([[0., 2.]])}
    n = {'path': torch.tensor([[2., 0.]])}
    l = ft.partial(training.triplet_loss, 0., lambda x: x)
    assert l(p, p, p) == torch.tensor([0.])
    assert l(p, p, n) == torch.tensor([0.])
    assert l(p, n, p) == torch.tensor([1.])
    assert l(p, n, n) == torch.tensor([0.])
    assert l(n, n, n) == torch.tensor([0.])

    l = ft.partial(training.triplet_loss, .5, lambda x: x)
    assert l(p, p, p) == torch.tensor([.5])
    assert l(p, p, n) == torch.tensor([0.])
    assert l(p, n, n) == torch.tensor([.5])
    assert l(n, n, n) == torch.tensor([.5])


def test_early_stopping():
    it_learning = training.make_is_learning(0, 0)
    assert it_learning({'val': {'loss': 0}}) == False
    #^ Not having patience means to stop immediately

    it_learning = training.make_is_learning(1, .1)
    assert it_learning({'val': {'loss': 3.00}}) == True
    assert it_learning({'val': {'loss': 2.91}}) == False
    #^ Validation loss doesn't decrease more than delta

    it_learning = training.make_is_learning(2, 0)
    assert it_learning({'val': {'loss': 3}}) == True
    assert it_learning({'val': {'loss': 2}}) == True
    assert it_learning({'val': {'loss': 1}}) == True
    assert it_learning({'val': {'loss': 2}}) == True
    assert it_learning({'val': {'loss': 0}}) == True
    assert it_learning({'val': {'loss': 1}}) == True
    assert it_learning({'val': {'loss': 2}}) == False
    #^ Patience over multiple losses

