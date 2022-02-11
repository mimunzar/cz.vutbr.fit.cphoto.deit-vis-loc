#!/usr/bin/env python3

import functools as ft
import operator  as op

import pytest
import torch

import src.deit_vis_loc.training as training


def test_make_triplets():
    queries_meta = {
        'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
    }
    triplets = training.make_triplets(queries_meta, queries_meta.keys())
    assert set(triplets('foo')) == {('foo', 'foo_p', 'foo_n')}
    with pytest.raises(KeyError): triplets('baz')

    queries_meta = {
        'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
        'bar': {'positive': {'bar_p'}, 'negative': {'bar_n'}},
    }
    triplets = training.make_triplets(queries_meta, queries_meta.keys())
    assert set(triplets('foo')) == {
            ('foo', 'foo_p', 'foo_n'),
            ('foo', 'foo_p', 'bar_n'),
            ('foo', 'foo_p', 'bar_p'),
        }
    assert set(triplets('bar')) == {
            ('bar', 'bar_p', 'bar_n'),
            ('bar', 'bar_p', 'foo_n'),
            ('bar', 'bar_p', 'foo_p'),
        }


def test_iter_triplets():
    queries_meta = {
        'foo': {'positive': {'foo_p'}, 'negative': {'foo_n'}},
        'bar': {'positive': {'bar_p'}, 'negative': {'bar_n'}},
    }
    assert list(training.iter_triplets(queries_meta, [])) == []
    assert list(training.iter_triplets(queries_meta, ['foo'])) == [
            {'anchor': 'foo', 'positive': 'foo_p', 'negative': 'foo_n'},
        ]
    triplets = training.iter_triplets(queries_meta, ['foo', 'bar'])
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
    loss = ft.partial(training.triplet_loss, 0, lambda x: x)
    assert loss({'anchor': z, 'positive': z, 'negative': z}) == torch.tensor([[0.]])
    assert loss({'anchor': z, 'positive': z, 'negative': o}) == torch.tensor([[0.]])
    assert loss({'anchor': z, 'positive': o, 'negative': z}) == torch.tensor([[1.]])
    assert loss({'anchor': z, 'positive': o, 'negative': o}) == torch.tensor([[0.]])
    assert loss({'anchor': o, 'positive': o, 'negative': o}) == torch.tensor([[0.]])

    loss = ft.partial(training.triplet_loss, 0.5, lambda x: x)
    assert loss({'anchor': z, 'positive': z, 'negative': z}) == torch.tensor([[0.5]])
    assert loss({'anchor': z, 'positive': z, 'negative': o}) == torch.tensor([[0.]])
    assert loss({'anchor': z, 'positive': o, 'negative': z}) == torch.tensor([[1.5]])
    assert loss({'anchor': z, 'positive': o, 'negative': o}) == torch.tensor([[0.5]])
    assert loss({'anchor': o, 'positive': o, 'negative': o}) == torch.tensor([[0.5]])


def test_iter_triplet_loss():
    z       = torch.zeros(1, 1)
    o       = torch.ones (1, 1)
    loss_it = training.make_iter_triplet_loss(0.5, lambda x: x, lambda _, q_it: q_it)
    assert list(loss_it({}, [])) == []
    assert list(loss_it({}, [
            {'anchor': z, 'positive': z, 'negative': z},
            {'anchor': z, 'positive': z, 'negative': o},
            {'anchor': z, 'positive': o, 'negative': z},
            {'anchor': z, 'positive': o, 'negative': o},
            {'anchor': o, 'positive': o, 'negative': o},
        ])) == [torch.tensor([[0.5]]),
                torch.tensor([[0.]]),
                torch.tensor([[1.5]]),
                torch.tensor([[0.5]]),
                torch.tensor([[0.5]])]


def test_early_stopping():
    it_learning = training.make_is_learning(0, 0)
    assert it_learning({'val': 0}) == False
    #^ Not having patience means to stop immediately

    it_learning = training.make_is_learning(1, .1)
    assert it_learning({'val': 3.00}) == True
    assert it_learning({'val': 2.91}) == False
    #^ Validation loss doesn't decrease more than delta

    it_learning = training.make_is_learning(2, 0)
    assert it_learning({'val': 3}) == True
    assert it_learning({'val': 2}) == True
    assert it_learning({'val': 1}) == True
    assert it_learning({'val': 2}) == True
    assert it_learning({'val': 0}) == True
    assert it_learning({'val': 1}) == True
    assert it_learning({'val': 2}) == False
    #^ Patience over multiple losses


def test_iter_test_pairs():
    queries_meta = {
        'q_1': {'positive': {'p_1'}, 'negative': {'n_1'}},
        'q_2': {'positive': {'p_2'}, 'negative': {'n_2'}},
    }
    assert list(training.iter_test_pairs(queries_meta, [])) == []
    assert list(training.iter_test_pairs(queries_meta, ['q_1'])) == [
        ('q_1', {('q_1', 'n_1'), ('q_1', 'p_1')})
    ]
    assert list(training.iter_test_pairs(queries_meta, ['q_1', 'q_2'])) == [
        ('q_1', {('q_1', 'n_1'), ('q_1', 'n_2'), ('q_1', 'p_1'), ('q_1', 'p_2')}),
        ('q_2', {('q_2', 'n_1'), ('q_2', 'n_2'), ('q_2', 'p_1'), ('q_2', 'p_2')}),
    ]

