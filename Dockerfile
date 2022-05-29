from  nvidia/cuda:11.4.0-devel-ubuntu20.04

# dont ask questions during installation
ENV DEBIAN_FRONTEND="noninteractive"

# update apt
RUN apt clean && apt update && apt-get install -y ca-certificates build-essential software-properties-common

# Install recent CMake.
RUN apt install -y wget && \
    wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1-linux-x86_64.sh && \
    chmod +x cmake-3.23.1-linux-x86_64.sh && \
    mkdir cmake && \
    ./cmake-3.23.1-linux-x86_64.sh --skip-license --prefix="$(pwd)/cmake" && \
    ln -s $(pwd)/cmake/bin/cmake /usr/local/bin/cmake

# Install recent GCC.
RUN echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 0E98404D386FA1D9 && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 648ACFD622F3D138 && \
    apt update -y && \
    apt install -y libc6-dev libstdc++-10-dev gcc-10 g++-10 && \
    ln -s $(which gcc-10) /usr/local/bin/gcc && \
    ln -s $(which g++-10) /usr/local/bin/g++

RUN apt install -y git wget make ffmpeg python3.9-dev python3.9-distutils

# check versions
RUN python3 --version
RUN python3.9 --version

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py

RUN pip3 install --upgrade pip && pip3 install --no-cache-dir numpy


# install scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
  libatlas-base-dev \
  gfortran \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/ \
  && pip3 install --no-cache-dir scipy


# install scikit-learn, scikit-image, pandas
RUN pip3 install --no-cache-dir \
    scikit-learn \
    scikit-image \
    pandas

RUN pip3 install torch==1.11.0 torchvision==0.12.0 svgwrite svgpathtools cssutils numpy torch-tools visdom

RUN git clone https://github.com/BachiLi/diffvg.git\
 && cd diffvg\
 && git submodule update --init --recursive\
 && DIFFVG_CUDA=1 CC=gcc-10 CXX=g++-10 python3.9 setup.py install --user

RUN apt-get -y update && \
    apt-get -y install vim

RUN pip3 install pytorch_lightning==0.9.0 opencv-python kornia==0.1.4.post2 torch_optimizer click test-tube 

