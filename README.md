# Installation
There are two methods to install the environment which is needed for following
the NIFTy tutorial. First, install it regularly into the home folder of your
machine. Second, use a Docker container. The advantage of the second approach is
that it is going to work without any problems most likely. Its downsides are:
Root access to start the Docker container (security implications), a full Docker
installation (security implications), you need to be familiar with Docker in
order to get your results out of the container.

## Standard

- Install git, python3 and python3-pip, python3-scipy, python3-matplotlib. On
  Debian-based systems possibly:

```
# apt-get update
# apt-get install git python3-pip python3-scipy python3-matplotlib 
```

- Install our Fouier transform package `pypocketfft`:

```
$ pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft.git
```

- Clone this repository:

```
$ git clone --recursive https://gitlab.mpcdf.mpg.de/ift/nifty_tutorial.git
```

- Install `nifty`:

```
$ cd nifty_tutorial/nifty
$ python3 nifty_tutorial/nifty/setup.py install -f
```

## Docker

- Build the image:

```
$ git clone https://gitlab.mpcdf.mpg.de/ift/nifty_tutorial.git
# docker build -t niftytutorial .
```

- Start a container and mount your local folder `/mnt` into the folder `/mnt` in
  the container. Do this only if you know what you are doing!

```
# docker run -v /mnt:/mnt -it niftytutorial
```

## Double-check the installation

Run the scripts `*_solution.py` from the repository, check that they run through
and look at the output plots.

