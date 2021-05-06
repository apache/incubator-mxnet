---
layout: page
title: Ubuntu Setup
action: Get Started
action_url: /get_started
permalink: /get_started/ubuntu_setup
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Installing MXNet from source on Ubuntu

The following installation instructions are for building MXNet from source on
computers running **Ubuntu 16.04** or higher. For instructions to build MXNet
from source on other platforms, see the general [Build From Source
guide](build_from_source).

Instead of building from source, you can install a binary version of MXNet. For
that, please follow the information at [Get Started](get_started).

Building MXNet from source is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. (optional) Install the supported language-specific packages for MXNet.

## Contents

* [CUDA Dependencies](#cuda-dependencies)
* [Build the MXNet shared library from source](#build-mxnet-from-source)
* [Install Language Packages](#installing-language-packages-for-mxnet)
    * [R](#install-the-mxnet-package-for-r)
    * [Julia](#install-the-mxnet-package-for-julia)
    * [Scala](#install-the-mxnet-package-for-scala)
    * [Java](#install-the-mxnet-package-for-java)
    * [Perl](#install-the-mxnet-package-for-perl)
  * [Contributions](#contributions)
  * [Next Steps](#next-steps)

<hr>

## CUDA Dependencies

If you plan to build with GPU, you need to set up the environment for CUDA and cuDNN.

First, download and install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit). CUDA 9.2 is recommended.

Then download [cuDNN 7.1.4](https://developer.nvidia.com/cudnn).

Unzip the file and change to the cuDNN root directory. Move the header and libraries to your local CUDA Toolkit folder:

```bash
tar xvzf cudnn-9.2-linux-x64-v7.1
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig
```

<hr>

## Build the MXNet shared library from source

It is recommended that you review the general [build from source](build_from_source) instructions before continuing.

On Ubuntu versions 16.04 or later, you need the following dependencies:

**Step 1:** Install prerequisite packages.

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ninja-build ccache libopenblas-dev libopencv-dev cmake
```

Instead of `libopenblas-dev` you may also choose a different math library.
Further information is provided in the source guide's [Math Library
Selection](build_from_source#math-library-selection) section.

`libopencv-dev` is an optional dependency. You can delete it from above `apt-get
install` line and build MXNet without OpenCV support by setting `USE_OPENCV` to
`OFF` in the configuration file described below.

Note: CMake 3.13 or higher is required. If you are running an older version of
CMake, you will see an error message like `CMake 3.13 or higher is required. You
are running version 3.10.2` during Step 3. In that case please update CMake on
your system. You can download and install latest CMake from https://cmake.org or
via `pip3` with `pip3 install --user --upgrade "cmake>=3.13.2"`. If `pip3` is
not available on your system, you can install it via `sudo apt-get install -y
python3-pip`. After installing cmake with `pip3`, it is usually available at
`~/.local/bin/cmake` or directly as `cmake`.

**Step 2:** Download MXNet sources and configure

Clone the repository:

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet
cp config/linux.cmake config.cmake  # or config/linux_gpu.cmake for build with CUDA
```

Please edit the config.cmake file based on your needs. The file contains a
series of `set(name value CACHE TYPE "Description")` entries. You can change the
values of the respective `value`.

For a GPU-enabled build make sure you have installed the [CUDA dependencies
first](#cuda-dependencies)). When building a GPU-enabled build on a machine
without GPU, MXNet build can't autodetect your GPU architecture and will target
all available GPU architectures. Please set the `MXNET_CUDA_ARCH` variable in
`config.cmake` to your desired cuda architecture to speed up the build.

To (optionally) build with MKL math library, please install MKL first based on
the guide in [Math Library Selection](build_from_source#math-library-selection).

**Step 3:** Build MXNet core shared library.

```bash
rm -rf build
mkdir -p build && cd build
cmake ..
cmake --build .
```

Specify `cmake --build . --parallel N` to set the number of parallel compilation
jobs. Default is derived from CPUs available.

After a successful build, you will find the `libmxnet.so` in the `build` folder
in your MXNet project root. `libmxnet.so` is required to install language
bindings described in the next section.

<hr>


## Installing Language Packages for MXNet

After you have installed the MXNet core library. You may install MXNet interface
packages for the programming language of your choice:
- [Python](#install-mxnet-for-python)
- [C++](#install-the-mxnet-package-for-c&plus;&plus;)
- [Clojure](#install-the-mxnet-package-for-clojure)
- [Julia](#install-the-mxnet-package-for-julia)
- [Perl](#install-the-mxnet-package-for-perl)
- [R](#install-the-mxnet-package-for-r)
- [Scala](#install-the-mxnet-package-for-scala)
- [Java](#install-the-mxnet-package-for-java)

<hr>

### Install MXNet for Python

To install the MXNet Python binding navigate to the root of the MXNet folder then run the following:

```bash
cd python
pip install --user -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means
that if you edit the source files, these changes will be reflected in the
package installed.

#### Optional Python Packages

You may optionally install ```graphviz``` library that is used for visualizing network graphs you build on MXNet. You
may also install [Jupyter Notebook](http://jupyter.readthedocs.io/) which is used for running MXNet tutorials and
examples.

```bash
pip install --user graphviz==0.8.4 jupyter
```
<hr>


### Install the MXNet Package for C++

Refer to the [C++ Package setup guide](c_plus_plus).
<hr>


### Install the MXNet Package for Clojure

Refer to the [Clojure setup guide](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package).
<hr>


### Install the MXNet Package for Julia

#### Install Julia
The package available through `apt-get` is old and not compatible with the latest version of MXNet.
Fetch the latest version (1.0.3 at the time of this writing).

```bash
wget -qO julia-10.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.3-linux-x86_64.tar.gz
```

Place the extracted files somewhere like a julia folder in your home dir.

```bash
mkdir ~/julia
mv julia-10.tar.gz ~/julia
cd ~/julia
tar xvf julia-10.tar.gz
```

Test Julia.

```bash
cd julia-1.0.3/bin
julia -e 'using InteractiveUtils; versioninfo()'
```

If you're still getting the old version, remove it.

```bash
sudo apt remove julia
```

Update your PATH to have Julia's new location. Add this to your `.zshrc`, `.bashrc`, `.profile` or `.bash_profile`.

```bash
export PATH=~/julia/julia-1.0.3/bin:$PATH
```

Validate your PATH.

```bash
echo $PATH
```

Validate Julia works and is the expected version.

```bash
julia -e 'using InteractiveUtils; versioninfo()'
```

#### Setup Your MXNet-Julia Environment

**For each of the following environment variables, add the commands to your `.zshrc`, `.bashrc`, `.profile`
or `.bash_profile` to make them persist.**

Create a `julia-depot` folder and environment variable.

```bash
mkdir julia-depot
export JULIA_DEPOT_PATH=$HOME/julia/julia-depot
```

To use the Julia binding with an existing `libmxnet` installation, set the `MXNET_HOME` environment variable to the
MXNet source root. For example:

```bash
export MXNET_HOME=$HOME/incubator-mxnet
```

Now set the `LD_LIBRARY_PATH` environment variable to where `libmxnet.so` is found. If you can't find it, you might
have skipped the building MXNet step. Go back and [build MXNet](#build-the-shared-library) first. For example:

```bash
export LD_LIBRARY_PATH=$HOME/incubator-mxnet/lib:$LD_LIBRARY_PATH
```

Verify the location of `libjemalloc.so` and set the `LD_PRELOAD` environment variable.
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so:$LD_PRELOAD
```

With all of these updates, here's an example of what you might want to have in your `.zshrc`, `.bashrc`, `.profile`
or `.bash_profile`.

```
export PATH=$HOME/bin:$HOME/.local/bin:$HOME/julia/julia-1.0.3/bin:$PATH
export JULIA_DEPOT_PATH=$HOME/julia/julia-depot
export MXNET_HOME=$HOME/incubator-mxnet
export LD_LIBRARY_PATH=$HOME/incubator-mxnet/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so:$LD_PRELOAD
```

Install MXNet with Julia:

```bash
julia --color=yes --project=./ -e \
	  'using Pkg; \
	   Pkg.develop(PackageSpec(name="MXNet", path = joinpath(ENV["MXNET_HOME"], "julia")))'
```

For more details about installing and using MXNet with Julia, see the
[MXNet Julia documentation]({{'/api/julia'|relative_url}}).

<hr>

### Install the MXNet Package for Perl

Before you build MXNet for Perl from source code, you must complete [building the shared library](#build-the-shared-library).
After you build the shared library, run the following command from the MXNet source root directory to build the MXNet
Perl package:

```bash
sudo apt-get install libmouse-perl pdl cpanminus swig libgraphviz-perl
cpanm -q -L "${HOME}/perl5" Function::Parameters Hash::Ordered PDL::CCS

MXNET_HOME=${PWD}
export LD_LIBRARY_PATH=${MXNET_HOME}/lib
export PERL5LIB=${HOME}/perl5/lib/perl5

cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
perl Makefile.PL INSTALL_BASE=${HOME}/perl5
make install

cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
perl Makefile.PL INSTALL_BASE=${HOME}/perl5
make install

cd ${MXNET_HOME}/perl-package/AI-MXNet/
perl Makefile.PL INSTALL_BASE=${HOME}/perl5
make install
```
<hr>


### Install the MXNet Package for R

Before you build MXNet for R from source code, you must complete
[building the shared library](#build-the-shared-library).

**Minimum Requirements**
1. [GCC 4.8](https://gcc.gnu.org/gcc-4.8/) or later to compile C++ 11.
2. [GNU Make](https://www.gnu.org/software/make/)

<br/>

**Build the MXNet core shared library**

**Step 1** Install R, cran-devtools and dependencies

To install R and the devtools, run

```bash
sudo apt-get update
sudo apt-get install -y r-base-core r-cran-devtools libcairo2-dev libxml2-dev libtiff-dev
```

`libxml2-dev` is required for the `roxygen2` dependency and `libcairo2-dev` is
required for the suggested `imager` dependency.

To generate documentation, it is also required to install `roxygen2`.

```bash
R
> install.packages("roxygen2")
> Would you like to use a personal library instead?  (y/n) y
> Would you like to create a personal library ... to install packages into?  (y/n) y
```

Note: To successfully complete the next step, you need a personal R library. If
you were able to run `install.packages("roxygen2")` above, you either had
already, or you have successfully created a personal library just now.

**Step 2** Make and install the MXNet-R bindings.

```bash
make -f R-package/Makefile rpkg
```

#### Verify MXNet-R Installation

You can verify your MXNet-R installation as follows:

```bash
R
```

At the R prompt enter the following:

```r
library(mxnet)
a <- mx.nd.ones(c(2,3), ctx = mx.cpu())
b <- a * 2 + 1
b
```

You should see the following output:

```
     [,1] [,2] [,3]
[1,]    3    3    3
[2,]    3    3    3
> quit()
```
<hr>


### Install the MXNet Package for Scala

To use the MXNet-Scala package, you can acquire the Maven package as a dependency.

Further information is in the [MXNet-Scala Setup Instructions](scala_setup).

If you use IntelliJ or a similar IDE, you may want to follow the
[MXNet-Scala on IntelliJ tutorial]({{'/api/scala/docs/tutorials/mxnet_scala_on_intellij'|relative_url}}) instead.
<hr>

### Install the MXNet Package for Java

To use the MXNet-Java package, you can acquire the Maven package as a dependency.

Further information is in the [MXNet-Java Setup Instructions](java_setup).

If you use IntelliJ or a similar IDE, you may want to follow the
[MXNet-Java on IntelliJ tutorial]({{'/api/java/docs/tutorials/mxnet_java_on_intellij'|relative_url}}) instead.
<hr>

## Contributions

You are more than welcome to contribute easy installation scripts for other operating systems and programming languages.
See the [community contributions page]({{'/community/contribute'|relative_url}}) for further information.

## Next Steps

* [Tutorials]({{'/api'|relative_url}})
* [How To]({{'/api/faq/add_op_in_backend'|relative_url}})
* [Architecture]({{'/api/architecture/overview'|relative_url}})


<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.1.0/css/all.css" integrity="sha384-lKuwvrZot6UHsBSfcMvOkWwlCMgc0TaWr+30HWe3a4ltaBwTZhyTEggF5tJv8tbt" crossorigin="anonymous">
