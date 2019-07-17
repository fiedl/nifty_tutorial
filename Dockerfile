FROM debian:testing-slim

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    # Needed for setup
    git python3-pip \
    # Packages needed for NIFTy
    python3-scipy \
    # Optional NIFTy dependencies
    python3-matplotlib \
  && pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft.git@c959e87dd69918fd63ff971fa2eebea99388a43c \
  && rm -rf /var/lib/apt/lists/*

# Set matplotlib backend
ENV MPLBACKEND agg
