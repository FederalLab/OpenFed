FROM hub.byted.org/base/lab.cuda.devel:10.2.89-1

ARG https_proxy=http://bj-rd-proxy.byted.org:3128
ARG no_proxy=apt.byted.org,mirrors.byted.org,bytedpypi.byted.org

# Install some system pre-request
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        build-essential \
        libopenmpi-dev \
        openmpi-bin \
        openssh-client \
        openssh-server \
        libgl1-mesa-glx \
        cmake \
        pkg-config \
        autoconf \
        libtool \
        automake \
        vim \
        git \
        sudo \
        tree \
        wget \
        curl \
        ca-certificates \
        unzip \
        iputils-ping \
        net-tools \
        tmux \
        htop \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# SSH related
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new
RUN echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new
RUN mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Set tos host ip (except for region China-North-LF)
ARG TOSV_HOST_IP=10.17.127.193

# Download toscli
RUN cd /tmp && echo "${TOSV_HOST_IP} tosv.byted.org" >> /etc/hosts && \
    wget http://tosv.byted.org/obj/bin/toscli -O toscli && \
    chmod a+x toscli && mv toscli /bin

# Set timezone for global regions
ARG TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# ============================ Python Dependencies ============================

# Upgrade pip
RUN pip3 install --no-cache-dir -U -i https://pypi.tuna.tsinghua.edu.cn/simple pip==20.2.3

# Install pytorch, horovod and apex
RUN pip3 install torch==1.5.1 torchvision==0.6.1
ARG HOROVOD_NCCL_HOME=/opt/tiger/cuda
ARG HOROVOD_CUDA_HOME=/opt/tiger/cuda
ARG HOROVOD_GPU_ALLREDUCE=NCCL
ARG HOROVOD_NCCL_LINK=SHARED
RUN pip3 install --no-cache-dir -i https://bytedpypi.byted.org/simple horovod==0.19.5
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install transformers with pytorch env only
RUN pip3 install --no-cache-dir transformers==3.5.1

# Install arnold dataloader
RUN pip3 install --no-cache-dir -U -i https://bytedpypi.byted.org/simple \
    byted-dataloader==0.2.6

# Install tensorflow for data loading and visualization
RUN pip3 install --no-cache-dir -U -i https://pypi.tuna.tsinghua.edu.cn/simple \
    tensorboardX \
    tensorflow-cpu==2.3.1

# Install mmcv-full
RUN pip3 install mmcv-full==latest+torch1.5.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html

# Install other PYPI packages
RUN pip3 install --no-cache-dir -U -i https://pypi.tuna.tsinghua.edu.cn/simple \
    setuptools \
    wheel \
    pyyaml \
    addict \
    thop \
    torchsummary \
    fvcore \
    terminaltables \
    matplotlib \
    opencv-python \
    mpi4py \
    pandas \
    tabulate \
    packaging

# Replace original Pillow with Pillow-simd
RUN apt-get update && apt-get install -y libjpeg-dev && \
    pip3 uninstall -y pillow && \
    CC="cc -mavx2" pip3 install --no-cache-dir -U -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall pillow-simd