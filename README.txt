Visual Localization using DeiT Transformer
==========================================

** The tool is still a work in progress **

Allows to train DeiT transformer for a visual localization [1].  The transformer
is trained on a sets of natural images and rendered views.  The task is to  find
a closest rendered view to a given image.  The given image than can be localized
by matching it's associated view with the position in the 3D terrain model  [2].


Installation
------------

To install the tool navigate to the project's folder  and  issue  the  following
command:

    conda env create environment.yml


Usage for Model Training [TODO]
-------------------------------

First download the GeoPose3K dataset [3].  The dataset should have the following
structure:
    .
    └── GeoPose3K_v2
        ├── database_segments
        ├── query_original_result
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        └── query_segments_result

Next specify model parameters in a JSON file. The parameters must have following
items:

    {
        "deit_model"    : "deit_tiny_patch16_224",
        "epochs"        : 10,
        "batch_size"    : 104,
        "triplet_margin": 0.2,
        "learning_rate" : 0.0001
    }


Then start trainining by executing the following command:

    python -m src.deit_vis_loc.train_model \
        --dataset_dir <PATH> \
        --save_dir    <PATH> \
        --params      <PATH>


Usage for Model Testing [TODO]
------------------------------


Tests
-----

  python -m pytest


Possible Improvements [TODO]
----------------------------


References
----------

  [1]: https://github.com/facebookresearch/deit
  [2]: http://cphoto.fit.vutbr.cz/crosslocate
  [3]: http://cphoto.fit.vutbr.cz/geoPose3K

