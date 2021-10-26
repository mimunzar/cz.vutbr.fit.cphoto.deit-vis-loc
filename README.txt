Visual Localization using DeiT Transformer
==========================================


Installation
------------

  conda env create environment.yml


Usage
---------

  python src/deit_vis_loc/main.py

  .
  └── GeoPose3K_v2
      ├── database_segments
      ├── query_original_result
      │   ├── test.txt
      │   ├── train.txt
      │   └── val.txt
      └── query_segments_result


Tests
-----

  python -m pytest


References
----------

  [1]: https://github.com/facebookresearch/deit

