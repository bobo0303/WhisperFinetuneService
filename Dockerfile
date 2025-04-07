FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
  
ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
WORKDIR /mnt  
  
# 當變成服務再改這個，然後用 compose
# WORKDIR /app  
# COPY . /app  

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 vim ffmpeg zip unzip htop screen tree build-essential gcc g++ make unixodbc-dev curl python3-dev python3-distutils wget libvulkan1 libfreeimage-dev libaio-dev \  
    && apt-get clean && rm -rf /var/lib/apt/lists  

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt  
  
ENV LC_ALL=C.UTF-8  
ENV LANG=C.UTF-8  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility  
ENV NVIDIA_VISIBLE_DEVICES=all  
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/llvm-10/lib:$LD_LIBRARY_PATH  
ENV PYTHONUTF8=1  

RUN rm /tmp/requirements.txt