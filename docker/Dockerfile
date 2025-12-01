# Base image
# FROM nvidia/cudagl:10.1-devel-ubuntu16.04
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

ENV USER=habitat_user

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.9 cmake=3.14.0

WORKDIR /home/${USER}

# Setup habitat-sim (maybe you can use stable version instead of v0.3.3)
RUN git clone --branch v0.3,3 https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate habitat; cd habitat-sim; pip install -r requirements.txt; python setup.py install --headless --bullet"

# Install habitat-lab (maybe you can use stable version instead of v0.3.3)
RUN git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab.git
RUN /bin/bash -c ". activate habitat; cd habitat-lab; pip install -e habitat-lab/"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

#
RUN conda run -n habitat python -m pip install notebook==7.3.3
RUN conda run -n habitat python -m pip install ipywidgets==8.1.5

ARG UID=1000
ARG GID=1000
RUN useradd -m ${USER} --uid=${UID} \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R $USER /home/${USER}
USER ${USER}

RUN echo 'source activate habitat' >> /home/${USER}/.bashrc

WORKDIR /home/${USER}/ws