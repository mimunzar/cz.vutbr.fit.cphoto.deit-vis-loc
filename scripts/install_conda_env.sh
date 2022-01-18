#!/bin/bash


function install_conda() {
    local installer="/tmp/miniconda.sh"
    local url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

    if [[ ! -e "${installer}" ]]; then
        curl -sSL ${url} -o ${installer}
        (( $? != 0 )) && echo "Download failed" && exit 1
    fi

    local install_dir=$1
    /bin/bash ${installer} -bfp ${install_dir} &> /dev/null
    (( $? != 0 )) && echo "Installation failed" && exit 1

    echo "${install_dir}/bin/conda"
}

function install_env() {
    local env_file=$1
    local conda_bin=$2

    /bin/bash -c "${conda_bin} update -yn base -c defaults conda" > /dev/null
    (( $? != 0 )) && echo "Update failed" && exit 1
    /bin/bash -c "${conda_bin} env create --force -f ${env_file}" > /dev/null
    (( $? != 0 )) && echo "Creating environment failed" && exit 1
    /bin/bash -c "${conda_bin} clean -ay" > /dev/null
    (( $? != 0 )) && echo "Cleanup failed" && exit 1

    echo $(head -n1 ${env_file} | cut -d' ' -f2)
}

info() {
    echo "[INFO][$(date +'%Y-%m-%dT%H:%M:%S')]: $*"
}

err() {
    echo "[ERROR][$(date +'%Y-%m-%dT%H:%M:%S')]: $*" >&2
}

function main() {
    local install_dir="$(readlink -f $1)"
    local env_file="$(readlink -f $2)"

    if [[ ! -e "${env_file}" ]]; then
        err "Failed to find Environment file (${env_file})" && exit 1
    fi

    info "Installing Miniconda to ${install_dir}"
    local conda_bin
    conda_bin="$(install_conda ${install_dir})"
    (( $? != 0 )) && err "Failed to install Miniconda (${conda_bin})" && exit 1
    conda_bin=$(readlink -f "${conda_bin}")
    info "Miniconda installed ${conda_bin}"

    info "Installing Environment from ${env_file}"
    local conda_env
    conda_env=$(install_env ${env_file} ${conda_bin})
    (( $? != 0 )) && err "Failed to install Environment (${conda_env})" && exit 1
    info "Environment installed ${conda_env}"
}

main "$@"

