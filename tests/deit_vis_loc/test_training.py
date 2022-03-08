#!/usr/bin/env python3

import functools as ft
import torch

import src.deit_vis_loc.training as training


def test_iter_im_pairs():
    assert list(map(set, training.iter_im_pairs({}, []))) == []
    meta = {'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}}}
    assert list(map(set, training.iter_im_pairs(meta, meta.keys()))) == [
        {('foo', 'foo_p'), ('foo', 'foo_n')}]
    meta = {
       'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
       'bar': {'positive': {'bar_p'}, 'negative': {'bar_n'}},
    }
    assert list(map(set, training.iter_im_pairs(meta, meta.keys()))) == [
        {('foo', 'foo_p'), ('foo', 'foo_n'), ('foo', 'bar_p'), ('foo', 'bar_n')},
        {('bar', 'bar_p'), ('bar', 'bar_n'), ('bar', 'foo_p'), ('bar', 'foo_n')}]


def test_iter_im_triplets():
    assert list(map(set, training.iter_im_triplets({}, []))) == []
    meta = {'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}}}
    assert list(map(set, training.iter_im_triplets(meta, meta.keys()))) == [
        {('foo', 'foo_p', 'foo_n')}]
    meta = {
       'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
       'bar': {'positive': {'bar_p'}, 'negative': {'bar_n'}},
    }
    assert list(map(set, training.iter_im_triplets(meta, meta.keys()))) == [
        {('foo', 'foo_p', 'foo_n'), ('foo', 'foo_p', 'bar_p'), ('foo', 'foo_p', 'bar_n')},
        {('bar', 'bar_p', 'bar_n'), ('bar', 'bar_p', 'foo_n'), ('bar', 'bar_p', 'foo_p')}]


def test_iter_hard_im_triplets():
    loss    = lambda s: int(s.endswith('_p')) + 2*int(s.startswith('foo'))
    tp_loss = lambda a, p, n: loss(a) + loss(p) + loss(n)
    meta    = {
       'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
       'bar': {'positive': {'bar_p'}, 'negative': {'bar_n'}},
    }
    iter_hard  = lambda n, meta: training.iter_hard_im_triplets(n, tp_loss, meta, meta.keys())

    assert list(map(list, iter_hard(3,   {}))) == []
    assert list(map(list, iter_hard(0, meta))) == [
            [],
            [],
        ]
    assert list(map(list, iter_hard(1, meta))) == [
            [('foo', 'foo_p', 'foo_n')],
            [('bar', 'bar_p', 'foo_p')],
        ]
    assert list(map(list, iter_hard(2, meta))) == [
            [
                ('foo', 'foo_p', 'foo_n'),
                ('foo', 'foo_p', 'bar_p'),
            ],
            [
                ('bar', 'bar_p', 'foo_p'),
                ('bar', 'bar_p', 'foo_n'),
            ],
        ]


def test_triplet_loss():
    p = torch.tensor([[0., 2.]])
    n = torch.tensor([[2., 0.]])
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
    assert it_learning({'vloss': 0}) == False
    #^ Not having patience means to stop immediately

    it_learning = training.make_is_learning(1, .1)
    assert it_learning({'vloss': 3.00}) == True
    assert it_learning({'vloss': 2.91}) == False
    #^ Validation loss doesn't decrease more than delta

    it_learning = training.make_is_learning(2, 0)
    assert it_learning({'vloss': 3}) == True
    assert it_learning({'vloss': 2}) == True
    assert it_learning({'vloss': 1}) == True
    assert it_learning({'vloss': 2}) == True
    assert it_learning({'vloss': 0}) == True
    assert it_learning({'vloss': 1}) == True
    assert it_learning({'vloss': 2}) == False
    #^ Patience over multiple losses

