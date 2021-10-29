#!/usr/bin/env python3

import src.deit_vis_loc.utils as utils


def test_partition():
    assert list(utils.partition(1, [])) == []
    assert list(utils.partition(4, [1, 2, 3])) == []
    assert list(utils.partition(3, [1, 2, 3])) == [[1, 2, 3]]
    assert list(utils.partition(2, [1, 2, 3])) == [[1, 2]]
    assert list(utils.partition(1, [1, 2, 3])) == [[1], [2], [3]]

