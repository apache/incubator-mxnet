#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if [ $# -lt 1 ]; then
    >&2 echo "Usage: build.sh <VARIANT>"
fi

export CURDIR=$PWD
export DEPS_PATH=$PWD/staticdeps
export VARIANT=$(echo $1 | tr '[:upper:]' '[:lower:]')
export PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')
export ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')

if [[ $VARIANT == darwin* ]]; then
    export VARIANT="darwin"
fi

NUM_PROC=1
if [[ ! -z $(command -v nproc) ]]; then
    NUM_PROC=$(nproc)
elif [[ ! -z $(command -v sysctl) ]]; then
    NUM_PROC=$(sysctl -n hw.ncpu)
else
    >&2 echo "Can't discover number of cores."
fi
export NUM_PROC
>&2 echo "Using $NUM_PROC parallel jobs in building."

if [[ $DEBUG -eq 1 ]]; then
    export ADD_MAKE_FLAG="-j $NUM_PROC"
else
    export ADD_MAKE_FLAG="--quiet -j $NUM_PROC"
fi
export MAKE="make $ADD_MAKE_FLAG"

if [[ $ARCH == 'aarch64' ]]; then
    export CC="gcc -fPIC -moutline-atomics"
    export CXX="g++ -fPIC -moutline-atomics"
    export PKG_CONFIG_PATH=$DEPS_PATH/lib/pkgconfig:$DEPS_PATH/lib64/pkgconfig:$DEPS_PATH/lib/aarch64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
else
    export CC="gcc -fPIC -mno-avx"
    export CXX="g++ -fPIC -mno-avx"
    export PKG_CONFIG_PATH=$DEPS_PATH/lib/pkgconfig:$DEPS_PATH/lib64/pkgconfig:$DEPS_PATH/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
fi
export FC="gfortran"
export CPATH=$DEPS_PATH/include:$CPATH

if [[ $PLATFORM == 'linux' && $VARIANT == cu* ]]; then
    source tools/setup_gpu_build_tools.sh $VARIANT $DEPS_PATH
fi

mkdir -p $DEPS_PATH

# Build Dependencies
source tools/dependencies/make_shared_dependencies.sh

# Copy LICENSE
mkdir -p licenses
cp tools/dependencies/LICENSE.binary.dependencies licenses/
cp NOTICE licenses/
cp LICENSE licenses/
cp DISCLAIMER-WIP licenses/


# Build mxnet
if [[ -z "$CMAKE_STATICBUILD" ]]; then
    source tools/staticbuild/build_lib.sh
else
    source tools/staticbuild/build_lib_cmake.sh
fi
