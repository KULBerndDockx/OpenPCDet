# OpenPCDet — Docker environment
# ------------------------------------------------------------------
# CUDA 11.3 + cuDNN 8 + Python 3.8 + PyTorch 1.10.0
# ------------------------------------------------------------------

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- system packages ------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake wget git vim fish \
        libsparsehash-dev \
        python3.8 python3.8-dev python3.8-venv python3-pip \
        libboost-all-dev && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- python packages ------------------------------------------------
RUN python -m pip install --no-cache-dir \
        numpy scipy matplotlib Cython && \
    python -m pip install --no-cache-dir \
        torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
        -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    python -m pip install --no-cache-dir \
        shapely fire pybind11 tensorboardX "protobuf>=3.0,<4.0" \
        scikit-image numba pillow

# ---- SparseConvNet (needed for SECOND imports, not PointPillars) ----
RUN git clone --depth 10 https://github.com/facebookresearch/SparseConvNet.git /tmp/SparseConvNet && \
    cd /tmp/SparseConvNet && python setup.py install && \
    rm -rf /tmp/SparseConvNet

# ---- numba CUDA env vars -------------------------------------------
ENV NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
ENV NUMBA_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
ENV NUMBA_LIBDEVICE=/usr/local/cuda/nvvm/libdevice

# ---- copy local codebase -------------------------------------------
COPY . /root/second.pytorch
ENV PYTHONPATH=/root/second.pytorch

VOLUME ["/root/data"]
VOLUME ["/root/model"]
WORKDIR /root/second.pytorch/second

ENTRYPOINT ["bash"]
