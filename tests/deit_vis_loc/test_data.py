#!/usr/bin/env python3

import src.deit_vis_loc.data as data

import math as ma


def test_is_circle_diff_close():
    assert data.is_circle_diff_close(ma.radians(30), ma.radians(0),  ma.radians(0))  == True
    assert data.is_circle_diff_close(ma.radians(30), ma.radians(30), ma.radians(0))  == True
    assert data.is_circle_diff_close(ma.radians(30), ma.radians(0),  ma.radians(30)) == True
    assert data.is_circle_diff_close(ma.radians(30), ma.radians(0),  ma.radians(31)) == False
    assert data.is_circle_diff_close(ma.radians(30), ma.radians(31), ma.radians(0))  == False
    #^ Similarity up to 30 degrees


def test_split_segments_by_yaw():
    list_of_segments = [
        {'camera_orientation': {'yaw': ma.radians(0)}},
        {'camera_orientation': {'yaw': ma.radians(30)}},
        {'camera_orientation': {'yaw': ma.radians(60)}},
    ]

    pos, neg = data.split_segments_by_yaw(ma.radians(0), ma.radians(30), list_of_segments)
    assert sorted(pos, key=lambda s: s['camera_orientation']['yaw']) == [
        {'camera_orientation': {'yaw': ma.radians(30)}},
    ]
    assert sorted(neg, key=lambda s: s['camera_orientation']['yaw']) == [
        {'camera_orientation': {'yaw': ma.radians(0)}},
        {'camera_orientation': {'yaw': ma.radians(60)}},
    ]

    pos, neg = data.split_segments_by_yaw(ma.radians(30), ma.radians(30), list_of_segments)
    assert sorted(pos, key=lambda s: s['camera_orientation']['yaw']) == [
        {'camera_orientation': {'yaw': ma.radians(0)}},
        {'camera_orientation': {'yaw': ma.radians(30)}},
        {'camera_orientation': {'yaw': ma.radians(60)}},
    ]
    assert sorted(neg, key=lambda s: s['camera_orientation']['yaw']) == []


