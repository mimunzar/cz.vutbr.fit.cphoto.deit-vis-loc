#!/bin/bash


INSTALL_DIR="$(readlink -f $1)"
source ${INSTALL_DIR}/etc/profile.d/conda.sh
ENV_FILE="$(readlink -f $2)"
conda activate $(head -n1 ${ENV_FILE} | cut -d' ' -f2)

shift 2
exec "$@"

