FROM debian:testing-slim

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    git python3-pip \
    python3-scipy \
    python3-matplotlib \
  && pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft.git@c959e87dd69918fd63ff971fa2eebea99388a43c \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://gitlab.mpcdf.mpg.de/ift/nifty_tutorial.git \
  && cd nifty_tutorial/nifty \
  && python3 setup.py install -f

ENV MPLBACKEND agg
WORKDIR /root
