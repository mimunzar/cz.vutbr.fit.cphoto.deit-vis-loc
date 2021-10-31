Visual Localization using DeiT Transformer
==========================================

** The tools is still a work in progress **

Allows to train DeiT transformer for a visual localization [1].  The transformer
is trained on a sets of natural images and rendered views.  The task is to  find
a closest rendered view to a given image.  The given image than can be localized
by matching it's associated view with the position in the 3D terrain model  [2].


Installation
------------

To install the tool navigate to the project's folder  and  issue  the  following
command:

    conda env create environment.yml


Usage
---------

    .
    └── GeoPose3K_v2
        ├── database_segments
        ├── query_original_result
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        └── query_segments_result


    python -m src.deit_vis_loc.main


Tests
-----

  python -m pytest


References
----------

  [1]: https://github.com/facebookresearch/deit
  [2]: http://cphoto.fit.vutbr.cz/crosslocate

