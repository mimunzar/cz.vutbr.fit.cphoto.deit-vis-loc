#!/bin/bash
#
#$ -S /bin/bash
#$ -N deit_vis_loc
#$ -l ram_free=16G,mem_free=16G
#$ -l gpu=1,gpu_ram=8G
#$ -l matylda1=3
#$ -q long.q@@gpu


PROJECT_ROOT="/mnt/matylda1/Locate/cz.vutbr.fit.cphoto.deit-vis-loc"
cd ${PROJECT_ROOT} || exit 1

ulimit -t $((14*86400))
mkdir -p output/
/bin/bash scripts/exec_in_conda_env.sh miniconda/ environment.yml \
    python -um src.deit_vis_loc.train_model \
        --dataset-dir  input/ \
        --metafile     input/queries_meta.json \
        --train-params input/train_params.json \
        --output-dir   output/ \
        --sge          &> output/"$(date +'%Y%m%dT%H%M%S').log"

