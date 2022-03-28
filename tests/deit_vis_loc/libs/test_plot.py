#!/usr/bin/env python3

import collections as cl
import pytest

import src.deit_vis_loc.libs.plot as plot


def test_im_and_n_closest_segments():
    segments = [
        {'segment': 's1', 'dist': 1, 'is_pos': True },
        {'segment': 's2', 'dist': 2, 'is_pos': False},
        {'segment': 's3', 'dist': 3, 'is_pos': False},
    ]
    assert list(plot.im_and_n_closest_segments(1, 'foo', []))       == [
            ('foo', 'black', 0)]
    assert list(plot.im_and_n_closest_segments(0, 'foo', segments)) == [
            ('foo', 'black', 0)]
    assert list(plot.im_and_n_closest_segments(1, 'foo', segments)) == [
            ('foo', 'black', 0), ('s1', 'green', 1)]
    assert list(plot.im_and_n_closest_segments(5, 'foo', segments)) == [
            ('foo', 'black', 0), ('s1', 'green', 1), ('s2', 'red', 2), ('s3', 'red', 3)]


def test_localization_percentile():
    with pytest.raises(Exception): plot.localization_percentile([])
    #^ Query image has to have positive segment
    assert plot.localization_percentile([
            {'dist': 1, 'is_pos': True},
        ]) == 100
    assert plot.localization_percentile([
            {'dist': 1, 'is_pos': False},
            {'dist': 2, 'is_pos': True },
            {'dist': 3, 'is_pos': False},
        ]) == 67
    assert plot.localization_percentile([
            {'dist': 1, 'is_pos': True },
            {'dist': 2, 'is_pos': False},
            {'dist': 3, 'is_pos': False},
        ]) == 34


def test_histogram():
    hist = plot.histogram([])
    assert hist[0] == 0
    assert hist[1] == 0

    hist = plot.histogram([1])
    assert hist[0] == 0
    assert hist[1] == 1

    hist = plot.histogram([1, 0, 1])
    assert hist[0] == 1
    assert hist[1] == 2


def test_iter_histogram_perc():
    hist = cl.defaultdict(lambda: 0, {})
    assert list(plot.iter_histogram_perc(hist, range(0))) == []
    assert list(plot.iter_histogram_perc(hist, range(1))) == []

    hist = cl.defaultdict(lambda: 0, {0: 1, 1: 2})
    assert list(plot.iter_histogram_perc(hist, range(0))) == []
    assert list(plot.iter_histogram_perc(hist, range(1))) == [33.33]
    assert list(plot.iter_histogram_perc(hist, range(2))) == [33.33, 66.67]

