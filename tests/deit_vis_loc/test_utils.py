#!/usr/bin/env python3

import src.deit_vis_loc.utils as utils


def test_partition():
    assert utils.partition(1, []) == []
    assert utils.partition(1, [1, 2, 3]) == [[1], [2], [3]]
    assert utils.partition(2, [1, 2, 3]) == [[1, 2], [3]]
    assert utils.partition(3, [1, 2, 3]) == [[1, 2, 3]]
    assert utils.partition(4, [1, 2, 3]) == [[1, 2, 3]]


