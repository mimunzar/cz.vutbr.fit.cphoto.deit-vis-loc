FROM centos:7

RUN yum -y update \
    && yum -y install curl bzip2

WORKDIR app/

COPY scripts/install_environment.sh scripts/
COPY environment.yml .
RUN /bin/bash scripts/install_environment.sh environment.yml

CMD ["/bin/bash"]

