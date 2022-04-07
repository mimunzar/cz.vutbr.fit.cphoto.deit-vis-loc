#!/usr/bin/env python3

import functools as ft

import src.deit_vis_loc.libs.util as util


IS_NUM      = lambda x: isinstance(x, (int, float))
IS_DICT     = lambda x: isinstance(x, dict)
IS_NONBLANK = lambda x: isinstance(x, str) and x.strip()
IS_POS      = lambda n: IS_NUM(n) and 0 < n
IS_INT      = lambda n: isinstance(n, int)
IS_POS_INT  = lambda n: IS_INT(n) and 0 < n
IS_MISSING  = lambda p, k: k not in p

POS_VALIDATORS = {
    'dist_m'      : util.make_validator('pos_dist_m must be positive int', IS_POS_INT),
    'dist_tol_m'  : util.make_validator('dist_tolerance_m must be positive int', IS_POS_INT),
    'yaw_deg'     : util.make_validator('yaw_deg must be positive int', IS_POS_INT),
    'yaw_tol_deg' : util.make_validator('yaw_tolerance_deg must be positive int', IS_POS_INT),
}

NEG_VALIDATORS = {
    'dist_m'      : util.make_validator('pos_dist_m must be positive int', IS_POS_INT),
    'dist_tol_m'  : util.make_validator('dist_tolerance_m must be positive int', IS_POS_INT),
}

VALIDATORS = {
    'batch_size' : util.make_validator('batch_size must be a positive int', IS_POS_INT),
    'deit_model' : util.make_validator('deit_model must be a non-blank', IS_NONBLANK),
    'input_size' : util.make_validator('input resolution must be a positive int', IS_POS_INT),
    'max_epochs' : util.make_validator('max_epochs must be a positive int', IS_POS_INT),
    'margin'     : util.make_validator('margin must be positive', IS_POS),
    'lr'         : util.make_validator('learning rate must be positive number', IS_POS),
    'patience'   : util.make_validator('patience must be positive int', IS_POS_INT),
    'positives'  : util.make_validator('positives must be dictionary', IS_DICT),
    'negatives'  : util.make_validator('negatives must be dictionary', IS_DICT),
}

def parser(validators, params):
    missing = tuple(filter(ft.partial(IS_MISSING, params), validators.keys()))
    checker = util.make_checker(validators)
    if missing         : raise ValueError(f'Missing config ({", ".join(missing)})')
    if checker(params) : raise ValueError(f'Invalid config ({", ".join(checker(params))})')
    return params


def parse(params):
    parser(VALIDATORS, params)
    parser(POS_VALIDATORS, params['positives'])
    parser(NEG_VALIDATORS, params['negatives'])
    return params

