MXNet supports running on ARM devices, such as the Raspberry PI.

These instructions will walk through how to build MXNet for the Raspberry Pi and
install the Python bindings for the library.

You can do a cross compilation build on your local machine (faster) or a native
build on-device (slower, but more foolproof).

The complete MXNet library and its requirements can take almost 200MB of RAM,
and loading large models with the library can take over 1GB of RAM. Because of
this, we recommend running MXNet on the Raspberry Pi 3 or an equivalent device
that has more than 1 GB of RAM and a Secure Digital (SD) card that has at least
4 GB of free memory.

## Native build on the Raspberry Pi

To build MXNet directly on the Raspberry Pi device, you can mainly follow the
standard [Ubuntu setup]({{'/get_started/ubuntu_setup|relative_url}})
instructions. However, skip the step of copying the `config/linux.cmake` to
`config.cmake` and instead run the `cmake` in the "Build MXNet core shared
library" step as follows:


```
rm -rf build
mkdir -p build && cd build
cmake \
  -DUSE_SSE=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_BLAS=Open \
  -DUSE_OPENCV=ON \
  -DUSE_OPENMP=ON \
  -DUSE_SIGNAL_HANDLER=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja ..
ninja -j$(nproc)
```

Some compilation units require memory close to 1GB, so it's recommended that you
enable swap as explained below and be cautious about increasing the number of
jobs when building (-j). Executing these commands start the build process, which
can take up to a couple hours, and creates a file called `libmxnet.so` in the
build directory.

If you are getting build errors in which the compiler is being killed, it is
likely that the compiler is running out of memory (especially if you are on
Raspberry Pi 1, 2 or Zero, which have less than 1GB of RAM), this can often be
rectified by increasing the swapfile size on the Pi by editing the file
/etc/dphys-swapfile and changing the line CONF_SWAPSIZE=100 to
CONF_SWAPSIZE=1024, then running:

```
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start
free -m # to verify the swapfile size has been increased
```

## Cross-compiling on your local machine

### Obtaining the toolchain

You first need to setup the cross-compilation toolchain on your local machine.
On Debian based systems, you can install `crossbuild-essential-armel` to obtain
a cross-toolchain for the ARMv4T, 5T, and 6, `crossbuild-essential-armhf` ARMv7
architecture and `crossbuild-essential-arm64` for ARMv8 (also called aarch64).
See for example
[Wikipedia](https://en.wikipedia.org/wiki/Raspberry_Pi#Specifications) to
determine the architecture of your Raspberry PI devices. If none of the Debian
toolchains works for you, you may like to refer to
[toolchains.bootlin.com](https://toolchains.bootlin.com/) for a large number of
ready-to-use cross-compilation toolchains.

### Cross-compiling MXNet dependencies
Before compiling MXNet, you need to cross-compile MXNet's dependencies. At the
very minimum, you'll need OpenBLAS. You can cross-compile it as follows,
replacing the `CC=aarch64-linux-gnu-gcc` and `PREFIX=/usr/aarch64-linux-gnu`
based on your architecture:

```
git clone --recursive https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make NOFORTRAN=1 NO_SHARED=1 CC=aarch64-linux-gnu-gcc
make PREFIX=/usr/local/aarch64-linux-gnu NO_SHARED=1 install
```

If you would like to compile MXNet with OpenCV support, enabling various image
transformation related features, you also need to cross-compile OpenCV.

### Cross-compiling MXNet

Before you cross-compile MXNet, create a CMake toolchain file specifying all settings for your compilation. For example, `aarch64-linux-gnu-toolchain.cmake`:

```
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR "aarch64")
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu;/usr/local/aarch64-linux-gnu")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

`CMAKE_FIND_ROOT_PATH` should be a list of directories containing the
cross-compilation toolchain and MXNet's cross-compiled dependencies. If you use
a toolchain from the bootlin site linked above, you can find the respective
CMake toolchain file at `share/buildroot/toolchainfile.cmake`.

You can then cross-compile MXNet via

```
mkdir build; cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
  -DUSE_CUDA=OFF \
  -DSUPPORT_F16C=OFF \
  -DUSE_BLAS=Open \
  -DUSE_OPENCV=OFF \
  -DUSE_OPENMP=ON \
  -DUSE_LAPACK=OFF \
  -DUSE_SIGNAL_HANDLER=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja ..
ninja
cd ..
```

We would like to simplify this setup by integrating the Conan C++ dependency
manager. Please send an email to the MXNet development mailinglist or open an
issue on GitHub if you would like to help.

### Building the Python wheel

To build the wheel, you can follow the following process

```
export MXNET_LIBRARY_PATH=$(pwd)/build/libmxnet.so

cd python
python3 setup.py bdist_wheel


# Fix pathing issues in the wheel.  We need to move libmxnet.so from the data folder to the
# mxnet folder, then repackage the wheel.
WHEEL=`readlink -f dist/*.whl`
TMPDIR=`mktemp -d`
unzip -d ${TMPDIR} ${WHEEL}
rm ${WHEEL}
cd ${TMPDIR}
mv *.data/data/mxnet/libmxnet.so mxnet
zip -r ${WHEEL} .
cp ${WHEEL} ..
rm -rf ${TMPDIR}
```

We intend to fix the `setup.py` to avoid the repackaging step. If you would like
to help, please send an email to the MXNet development mailinglist or open an
issue on GitHub.


### Final remarks

You are now ready to run MXNet on your Raspberry Pi device. You can get started
by following the tutorial on [Real-time Object Detection with MXNet On The
Raspberry
Pi](https://mxnet.io/api/python/docs/tutorials/deploy/inference/wine_detector.html).

*Note - Because the complete MXNet library takes up a significant amount of the
Raspberry Pi's limited RAM, when loading training data or large models into
memory, you might have to turn off the GUI and terminate running processes to
free RAM.*
