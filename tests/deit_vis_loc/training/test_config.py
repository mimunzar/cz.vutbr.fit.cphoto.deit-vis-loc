#!/usr/bin/env python3

import pytest
import src.deit_vis_loc.training.config as config


def test_parse():
    params = {
        'lr'               : 1e-3,
        'batch_size'       : 5,
        'margin'           : 0.1,
        'max_epochs'       : 100,
        'mine_every_epoch' : 1,
        'positives'  : {
            'samples'     : 1,
            'dist_m'      : 100,
            'dist_tol_m'  : 10,
            'yaw_deg'     : 15,
            'yaw_tol_deg' : 1,
        },
        'negatives'  : {
            'samples'    : 5,
            'dist_m'     : 200,
            'dist_tol_m' : 10,
        }
    }
    assert config.parse(params) == params

    inv_params = {**params}
    del inv_params['lr']
    del inv_params['positives']
    with pytest.raises(ValueError): config.parse(inv_params)

    inv_params = {**params}
    inv_params['lr'] = 'foo'
    with pytest.raises(ValueError): config.parse(inv_params)

    inv_params = {**params}
    inv_params['lr'] = None
    with pytest.raises(ValueError): config.parse(inv_params)

    inv_params = {**params}
    del inv_params['positives']['dist_m']
    with pytest.raises(ValueError): config.parse(inv_params)

    inv_params = {**params}
    del inv_params['negatives']['dist_m']
    with pytest.raises(ValueError): config.parse(inv_params)

