#!/usr/bin/env python3

import functools as ft
import torch

import src.deit_vis_loc.training.locate as locate


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

