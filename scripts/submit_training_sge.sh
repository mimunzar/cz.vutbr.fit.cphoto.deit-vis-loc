#!/bin/bash
#
#$ -S /bin/bash
#$ -N deit_vis_loc
#$ -pe smp 2
#$ -l gpu=1,gpu_ram=8G
#$ -l ram_free=16G,mem_free=16G
#$ -l matylda1=2
#$ -q long.q@@gpu
#$ -R y

PROJECT_ROOT=/mnt/matylda1/Locate/cz.vutbr.fit.cphoto.deit-vis-loc

activate_env() {
    unset PYTHONHOME
    source miniconda/etc/profile.d/conda.sh
    conda activate $(head -n1 environment.yml | cut -d' ' -f2)
}

train_network() {
    ulimit -t $((14*86400))
    mkdir -p output/
    python -um src.deit_vis_loc.train_model \
        --dataset-dir  input/ \
        --params       input/params.json \
        --output-dir   output/ \
        --workers      ${NSLOTS} \
        --device       cuda \
        --sge &> output/"$(date +'%Y%m%dT%H%M%S').log"
}

cd ${PROJECT_ROOT} || exit 1
activate_env
train_network

