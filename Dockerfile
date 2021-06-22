FROM hub.byted.org/base/lab.cuda.devel:10.2.89-3

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
    python3-opencv=3.2.0+dfsg-6 \
    && \
    rm -rf /var/lib/apt/lists/*

# SSH related
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new
RUN echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new
RUN mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Set timezone for global regions
ARG TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# ============================ Python Dependencies ============================

# Upgrade pip
RUN pip3 install --no-cache-dir -U -i https://pypi.tuna.tsinghua.edu.cn/simple pip==20.2.3

# Install pytorch, horovod and apex
RUN pip3 install torch==1.9.0 torchaudio==0.9.0 torchvision==0.10.0
ARG HOROVOD_GPU_OPERATIONS=NCCL
ARG HOROVOD_NCCL_LINK=SHARED
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install arnold dataloader
RUN pip3 install --no-cache-dir -U -i https://bytedpypi.byted.org/simple \
    byted-dataloader==0.2.6 \
    bytedeuler==0.22.1

# Install other PYPI packages
RUN pip3 install --no-cache-dir -U \
    setuptools \
    wheel \
    pyyaml \
    thop \
    torchsummary \
    fvcore \
    terminaltables \
    matplotlib \
    mpi4py \
    pandas \
    tabulate \
    packaging \
    scipy \
    Pillow \
    shapely \
    ray==1.2.0 \
    bidict==0.21.2 \
    h5py==3.2.1 \
    loguru==0.5.3 \
    numpy==1.20.2 \
    prettytable==2.1.0 \
    psutil==5.8.0 \
    tensorboard==2.4.1 \
    tensorboard-plugin-wit==1.8.0 \
    tensorboardX==2.2 \
    torchmetrics==0.2.0

# Add Packing Dependency
RUN SCM_NAME=toutiao.videoarch.pyiam && \
    SCM_VERSION=1.0.0.19 && \
    SCM_PATH=/opt/tiger/pyiam && \
    (wget http://d.scm.byted.org/api/v2/download/ceph:${SCM_NAME}_${SCM_VERSION}.tar.gz -O /tmp/${SCM_NAME}.tar.gz -q || exit -1) && \
    (mkdir -p ${SCM_PATH} && tar -zxf /tmp/${SCM_NAME}.tar.gz -C ${SCM_PATH} || exit -1) && \
    rm -rf /tmp/${SCM_NAME}.tar.gz
RUN SCM_NAME=content.review.risk_predict && \
    SCM_VERSION=1.0.0.1849 && \
    SCM_PATH=/opt/tiger/risk_predict && \
    (wget http://d.scm.byted.org/api/v2/download/ceph:${SCM_NAME}_${SCM_VERSION}.tar.gz -O /tmp/${SCM_NAME}.tar.gz -q || exit -1) && \
    (mkdir -p ${SCM_PATH} && tar -zxf /tmp/${SCM_NAME}.tar.gz -C ${SCM_PATH} || exit -1) && \
    rm -rf /tmp/${SCM_NAME}.tar.gz
ENV WEBP_LIBRARY_PATH /opt/tiger/risk_predict

# Move OpenFed project to container
RUN mkdir -p /opt/tiger
# Manually add
ADD . /opt/tiger/OpenFed
# After open source
# RUN cd /opt/tiger && git clone git@github.com:FederalLab/OpenFed.git 

# Set working directory
WORKDIR /opt/tiger/OpenFed