FROM debian:testing-slim

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    git python3-pip \
    python3-scipy \
    python3-matplotlib \
  && pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft.git \
  && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND agg
WORKDIR /tutorial

