#!/usr/bin/env python3

import src.deit_vis_loc.plot as plot
import pytest

import collections as cl


def test_ims_of_query_with_n_segments():
    segments = [
        {'name': 's3', 'distance': 3, 'is_positive': False},
        {'name': 's1', 'distance': 1, 'is_positive': True},
        {'name': 's2', 'distance': 2, 'is_positive': False},
    ]
    assert list(plot.ims_of_query_with_n_segments(42, 'foo', [])) == [
            ('foo', 'black', 0)]
    assert list(plot.ims_of_query_with_n_segments(0,  'foo', segments)) == [
            ('foo', 'black', 0)]
    assert list(plot.ims_of_query_with_n_segments(1,  'foo', segments)) == [
            ('foo', 'black', 0), ('s1', 'green', 1)]
    assert list(plot.ims_of_query_with_n_segments(5,  'foo', segments)) == [
            ('foo', 'black', 0), ('s1', 'green', 1), ('s2', 'red', 2), ('s3', 'red', 3)]


def test_localization_percentile():
    with pytest.raises(Exception):
        plot.localization_percentile([])
    #^ Query has to always be associated with a positive segment

    assert plot.localization_percentile([
            {'distance': 1, 'is_positive': True},
        ]) == 100
    assert plot.localization_percentile([
            {'distance': 2, 'is_positive': True},
            {'distance': 1, 'is_positive': False},
        ]) == 100
    assert plot.localization_percentile([
            {'distance': 1, 'is_positive': True},
            {'distance': 2, 'is_positive': False},
            {'distance': 3, 'is_positive': False},
        ]) == 34


def test_percentiles_histogram():
    with pytest.raises(Exception):
        plot.running_localization_percentage([])

    result = plot.running_localization_percentage([1])
    assert 100 == len(result)
    assert 100 == result[0]
    assert all(r == 0 for r in result[1:])

    result = plot.running_localization_percentage([2, 1, 1])
    assert 100 == len(result)
    assert [66.67, 33.33] == result[0:2]
    assert all(r == 0 for r in result[2:])

    result = plot.running_localization_percentage([2, 1, 1, 3])
    assert 100 == len(result)
    assert [50, 25, 25] == result[0:3]
    assert all(r == 0 for r in result[3:])

