#!/usr/bin/env python3

import functools as ft
import operator  as op

import pytest
import torch

import src.deit_vis_loc.model as model


def test_make_qpn():
    meta = {
        'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
    }
    qpn  = model.make_qpn(meta, meta.keys())
    assert qpn('foo') == ({'foo'}, {'foo_p'}, {'foo_n'})
    with pytest.raises(KeyError): qpn('bar')

    meta = {
        'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
        'bar': {'positive': {'bar_p'}, 'negative': {'bar_n'}},
    }
    qpn  = model.make_qpn(meta, meta.keys())
    assert qpn('foo') == ({'foo'}, {'foo_p'}, {'bar_p', 'bar_n', 'foo_n'})
    assert qpn('bar') == ({'bar'}, {'bar_p'}, {'foo_p', 'foo_n', 'bar_n'})


def test_iter_triplets():
    meta = {
        'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
        'bar': {'positive': {'bar_p'}, 'negative': {'bar_n'}},
    }
    assert list(model.iter_triplets(meta, [])) == []
    assert list(model.iter_triplets(meta, ['foo'])) == [
            {'anchor': 'foo', 'positive': 'foo_p', 'negative': 'foo_n'},
        ]

    triplets = model.iter_triplets(meta, ['foo', 'bar'])
    assert sorted(triplets, key=op.itemgetter('anchor', 'negative')) == [
            {'anchor': 'bar', 'positive': 'bar_p', 'negative': 'bar_n'},
            {'anchor': 'bar', 'positive': 'bar_p', 'negative': 'foo_n'},
            {'anchor': 'bar', 'positive': 'bar_p', 'negative': 'foo_p'},
            {'anchor': 'foo', 'positive': 'foo_p', 'negative': 'bar_n'},
            {'anchor': 'foo', 'positive': 'foo_p', 'negative': 'bar_p'},
            {'anchor': 'foo', 'positive': 'foo_p', 'negative': 'foo_n'},
        ]


def test_triplet_loss():
    z = torch.zeros(1, 1)
    o = torch.ones (1, 1)
    loss = ft.partial(model.triplet_loss, lambda x: x, 0)
    assert loss({'anchor': z, 'positive': z, 'negative': z}) == torch.tensor([[0.]])
    assert loss({'anchor': z, 'positive': z, 'negative': o}) == torch.tensor([[0.]])
    assert loss({'anchor': z, 'positive': o, 'negative': z}) == torch.tensor([[1.]])
    assert loss({'anchor': z, 'positive': o, 'negative': o}) == torch.tensor([[0.]])
    assert loss({'anchor': o, 'positive': o, 'negative': o}) == torch.tensor([[0.]])

    loss = ft.partial(model.triplet_loss, lambda x: x, 0.5)
    assert loss({'anchor': z, 'positive': z, 'negative': z}) == torch.tensor([[0.5]])
    assert loss({'anchor': z, 'positive': z, 'negative': o}) == torch.tensor([[0.]])
    assert loss({'anchor': z, 'positive': o, 'negative': z}) == torch.tensor([[1.5]])
    assert loss({'anchor': z, 'positive': o, 'negative': o}) == torch.tensor([[0.5]])
    assert loss({'anchor': o, 'positive': o, 'negative': o}) == torch.tensor([[0.5]])


def test_iter_triplet_loss():
    z       = torch.zeros(1, 1)
    o       = torch.ones (1, 1)
    loss_it = model.make_iter_triplet_loss(lambda x: x, margin=0.5)
    assert list(loss_it([])) == []
    assert list(loss_it([
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


def test_early_stopping():
    is_trained = model.make_early_stoping(0, 0)
    assert is_trained(0) == True
    #^ Not having patience means to stop immediately

    is_trained = model.make_early_stoping(1, .1)
    assert is_trained(3.00) == False
    assert is_trained(2.91) == True
    #^ Validation loss doesn't decrease more than delta

    is_trained = model.make_early_stoping(2, 0)
    assert is_trained(3) == False
    assert is_trained(2) == False
    assert is_trained(1) == False
    assert is_trained(2) == False
    assert is_trained(0) == False
    assert is_trained(1) == False
    assert is_trained(2) == True
    #^ Patience over multiple losses


def test_gen_test_pairs():
    fake_rendered_segments = {
        "q_1": {"positive": {"p_1"}, "negative": {"n_1"}},
        "q_2": {"positive": {"p_2"}, "negative": {"n_2"}},
    }
    assert list(model.iter_test_pairs([], fake_rendered_segments)) == []
    assert list(model.iter_test_pairs(['q_1'], fake_rendered_segments)) == [
        ('q_1', [('q_1', 'p_1'), ('q_1', 'n_1')])
    ]
    assert list(model.iter_test_pairs(['q_1', 'q_2'], fake_rendered_segments)) == [
        ('q_1', [('q_1', 'p_1'), ('q_1', 'n_1'), ('q_1', 'p_2'), ('q_1', 'n_2')]),
        ('q_2', [('q_2', 'p_1'), ('q_2', 'n_1'), ('q_2', 'p_2'), ('q_2', 'n_2')]),
    ]

