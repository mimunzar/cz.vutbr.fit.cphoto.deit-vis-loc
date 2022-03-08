FROM conda/miniconda3

WORKDIR /app
COPY ./environment.yml ./
#^ Build environment first so changes to the source don't trigger reinstallation

RUN conda update -n base -c defaults conda \
    && conda env create -f environment.yml \
    && conda clean -ay \
    && conda init bash \
    && echo "conda activate $(head -n1 < environment.yml | cut -d' ' -f2)" >> ~/.bashrc

COPY ./src/ ./src/
ENTRYPOINT ["/usr/bin/env", "bash", "-lc"]

