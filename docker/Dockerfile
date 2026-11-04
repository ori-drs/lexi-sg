# =========================
# Stage 1: builder
# =========================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=lexisg

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libglu1-mesa-dev \
    libsm6 \
    libusb-1.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    openssh-client \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*


# miniconda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniconda.sh && \
    conda config --system --remove channels defaults || true && \
    conda config --system --remove-key default_channels || true && \
    conda config --system --remove-key custom_channels || true && \
    conda config --system --add channels conda-forge && \
    conda config --system --set channel_priority strict && \
    conda clean -afy

SHELL ["/bin/bash", "-lc"]

# create env + core deps
RUN conda create -y -n "${ENV_NAME}" python=3.11 pip && \
    conda install -y -n "${ENV_NAME}" -c conda-forge \
      cmake \
      eigen \
      ninja \
      pcl=1.12 \
      flann \
      pybind11 && \
    conda run -n "${ENV_NAME}" pip install --upgrade pip setuptools wheel

# python deps
COPY docker/requirements.lexisg.txt /tmp/requirements.lexisg.txt

# torch
RUN conda run -n "${ENV_NAME}" pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN conda run -n "${ENV_NAME}" pip install --no-cache-dir -r /tmp/requirements.lexisg.txt && \
    conda run -n "${ENV_NAME}" pip install "transformers==4.28.1" && \
    conda run -n "${ENV_NAME}" pip install "fairscale==0.4.13" && \
    conda run -n "${ENV_NAME}" pip install supervision==0.26.0

# copy source ONLY for build (not final image)
WORKDIR /build
COPY . /build

# patch GroundingDINO for PyTorch 2.7+ compatibility
RUN sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' \
        /build/thirdparty/Grounded-Segment-Anything/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h && \
    sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' \
        /build/thirdparty/Grounded-Segment-Anything/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu

# install packages (compiled parts)
RUN conda run -n "${ENV_NAME}" pip install --no-cache-dir --no-deps /build && \
    AM_I_DOCKER=true BUILD_WITH_CUDA=true TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0" conda run -n "${ENV_NAME}" pip install --no-cache-dir --no-build-isolation /build/thirdparty/Grounded-Segment-Anything/GroundingDINO && \
    conda run -n "${ENV_NAME}" pip install --no-cache-dir --no-build-isolation /build/thirdparty/Grounded-Segment-Anything/segment_anything && \
    conda run -n "${ENV_NAME}" pip install --no-cache-dir --no-build-isolation /build/thirdparty/map-anything && \
    conda run -n "${ENV_NAME}" pip install --no-cache-dir --no-build-isolation /build/thirdparty/recognize-anything && \
    conda run -n "${ENV_NAME}" pip install --no-cache-dir --no-build-isolation --no-deps /build/thirdparty/sam2 && \
    conda run -n "${ENV_NAME}" pip install --no-cache-dir /build/thirdparty/openlex3d && \
    conda run -n "${ENV_NAME}" pip install faiss-cpu && \
    conda run -n "${ENV_NAME}" pip check

# =========================
# Stage 2: final
# =========================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=lexisg

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CONDA_DIR}/envs/${ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH}

# minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libglu1-mesa-dev \
    libsm6 \
    libusb-1.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    openssh-client \
    curl \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

# copy conda env from builder
COPY --from=builder /opt/conda /opt/conda

# entrypoint
COPY docker/entrypoint.sh /usr/local/bin/lexisg-entrypoint.sh
RUN chmod +x /usr/local/bin/lexisg-entrypoint.sh

WORKDIR /workspace/lexi-sg

# runtime cache dirs
RUN mkdir -p /workspace/lexi-sg/.cache/huggingface \
             /workspace/lexi-sg/.cache/torch \
             /tmp/matplotlib /home/lexisg && \
    chmod 1777 /tmp/matplotlib /home/lexisg

ENV HOME=/home/lexisg
ENV USER=lexisg LOGNAME=lexisg
ENV PYTHONPATH=/workspace/lexi-sg

ENTRYPOINT ["/usr/local/bin/lexisg-entrypoint.sh"]
CMD ["/bin/bash"]
