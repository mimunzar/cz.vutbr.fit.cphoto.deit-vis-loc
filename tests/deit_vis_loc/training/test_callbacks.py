#!/usr/bin/env python3

import src.deit_vis_loc.training.callbacks as callbacks


def test_iter_add_axis():
    assert tuple(callbacks.iter_prependaxis(
        [42, 42, 42], 1)) == ((1, 2, 3), (42, 42, 42))
    assert tuple(callbacks.iter_prependaxis(
        [42, 42, 42], 2)) == ((1, 3, 5), (42, 42, 42))
    assert tuple(callbacks.iter_prependaxis(
        [42, 42, 42], 3)) == ((1, 4, 7), (42, 42, 42))


def test_is_miningepoch():
    assert callbacks.is_miningepoch(1, 1) == True
    assert callbacks.is_miningepoch(2, 1) == True

    assert callbacks.is_miningepoch(1, 2) == True
    assert callbacks.is_miningepoch(2, 2) == False
    assert callbacks.is_miningepoch(3, 2) == True
    assert callbacks.is_miningepoch(4, 2) == False

    assert callbacks.is_miningepoch(1, 3) == True
    assert callbacks.is_miningepoch(2, 3) == False
    assert callbacks.is_miningepoch(3, 3) == False
    assert callbacks.is_miningepoch(4, 3) == True

