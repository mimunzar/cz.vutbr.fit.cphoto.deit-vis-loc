#!/usr/bin/env python3


import src.deit_vis_loc.main as main


def test_partition():
    assert main.partition(1, []) == []
    assert main.partition(1, [1, 2, 3]) == [[1], [2], [3]]
    assert main.partition(2, [1, 2, 3]) == [[1, 2], [3]]
    assert main.partition(3, [1, 2, 3]) == [[1, 2, 3]]
    assert main.partition(4, [1, 2, 3]) == [[1, 2, 3]]

