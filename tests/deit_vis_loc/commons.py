#!/usr/bin/env python3

PRETRAINING_PARAMS =  {
    'positives': {
        'samples'     : 1,
        'dist_m'      : 0,
        'dist_tol_m'  : 1,
        'yaw_deg'     : 0,
        'yaw_tol_deg' : 1,
   },
   'negatives': {
       'samples'     : 5,
       'dist_m'      : 2000,
       'dist_tol_m'  : 10,
   }
}

SPARSE_PARAMS = {
    'positives': {
        'samples'     : 1,
        'dist_m'      : 20,
        'dist_tol_m'  : 1,
        'yaw_deg'     : 15,
        'yaw_tol_deg' : 1,
    },
    'negatives': {
        'samples'     : 5,
        'dist_m'      : 2000,
        'dist_tol_m'  : 10,
    }
}

