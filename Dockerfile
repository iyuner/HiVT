FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
LABEL maintainer="Kin Zhang <qingwen@kth.se>"
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install -y wget git curl rsync ssh htop pip

# install zsh
RUN apt update && apt install -y zsh tmux vim
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    printf "y\ny\ny\n\n" | bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kin-Zhang/Kin-Zhang/main/scripts/setup_ohmyzsh.sh)"

RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cmake && pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl && \
    pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl && \
    pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl && \
    pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl

# av 1.1 --> ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt update && apt install -y libgl1 libglib2.0-0
RUN git clone https://github.com/argoverse/argoverse-api.git && \
    cd argoverse-api && pip install . && \
    pip install numpy==1.20.3 && \
    pip uninstall torch_geometric && pip install torch_geometric==1.7.2

# need this data in packages to read
RUN cd /usr/local/lib/python3.8/dist-packages && wget https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/hd_maps.tar.gz && tar -xvf hd_maps.tar.gz

RUN cd /root && git clone https://github.com/ZikangZhou/HiVT.git && pip install pytorch-lightning==1.5.2 \
    && apt update && apt install -y python-is-python3
    
WORKDIR /root/HiVT
