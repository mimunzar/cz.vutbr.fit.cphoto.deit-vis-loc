Visual Localization With Transformers
=====================================

** The tool is still a work in progress **

Allows to train DeiT transformer for a visual localization [1].  The transformer
is trained on a sets of natural images and rendered views.  The task is to  find
a closest rendered view to a given image.  The given image than can be localized
by matching it's associated view with the position in the 3D terrain model  [2].


Installation
------------

To install the tool navigate to the project's folder  and  issue  the  following
command:

    conda env create -f environment.yml


Build Dataset
-------------

    python -um src.deit_vis_loc.data.make_dataset \
        --geopose-dir ${DATA_GEOPOSE} \
        --sparse-dir  ${DATA_SPARSE} \
        --output-dir  data/ \
        --dataset     pretraining \
        --modality    silhouettes \
        --input-size  224


Train Model
-----------

  python -um src.deit_vis_loc.train_model \
        --data-dir     data/ \
        --dataset      pretraining \
        --modality     silhouettes \
        --input-size   224 \
        --params       params/pretraining.json \
        --model-name   deit_tiny_patch16_224 \
        --device       cuda \
        --gpu-imcap    300 \
        --output-dir   output/ \


Run Tests
---------

  python -m pytest


References
----------

  [1]: https://github.com/facebookresearch/deit
  [2]: http://cphoto.fit.vutbr.cz/crosslocate
  [3]: http://cphoto.fit.vutbr.cz/geoPose3K

