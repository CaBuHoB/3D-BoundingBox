FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y wget libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Copy files
WORKDIR /project
COPY . . 

# Create conda env && clear cache folder ~8.5GB
RUN conda env create --prefix ./env --file environment.yml \
    && conda clean -y --force-pkgs-dir

# Dowload pretrained weights for vgg
RUN ./env/bin/python -c 'from torchvision.models import vgg; vgg.vgg19_bn(pretrained=True)'

# Change entrypoint
ENTRYPOINT ["./scripts/run_docker.sh"]