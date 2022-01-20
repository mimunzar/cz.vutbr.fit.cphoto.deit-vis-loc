FROM centos:7

RUN yum -y update \
    && yum -y install curl bzip2

WORKDIR app/

COPY scripts/install_conda_env.sh scripts/
COPY environment.yml .
RUN /bin/bash scripts/install_conda_env.sh miniconda/ environment.yml

COPY scripts/exec_in_conda_env.sh scripts/
COPY src/ src/
ENTRYPOINT ["/bin/bash", "scripts/exec_in_conda_env.sh", "miniconda/", "environment.yml"]

