#!/bin/bash

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

set -eo pipefail

# This script generates wheel for macOS

#Install brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

#install developement tools
brew install nasm
brew install automake
brew install libtool
brew install ninja
brew install pkg-config
python -m pip install contextvars
python -m pip install numpy
python -m pip install requests

#clone MXNet
cd $HOME
git clone -b v1.x --single-branch https://github.com/apache/incubator-mxnet.git
cd $HOME/incubator-mxnet/
git submodule update --init --recursive

# set ENV variables and build statically
export IS_RELEASE=True
export mxnet_variant=cpu
echo $(git rev-parse HEAD) >> python/mxnet/COMMIT_HASH
mkdir -p $HOME/pip_build/mxnet-build
cp -r * $HOME/pip_build/mxnet-build
export CODE_DIR=$(pwd)
cd $HOME/pip_build/mxnet-build
CMAKE_STATICBUILD=1 ./tools/staticbuild/build.sh $mxnet_variant pip

# Package wheel
cd $HOME/pip_build
cp -r mxnet-build/tools/pip/* .
python setup.py bdist_wheel

# copy wheel to $HOME/pip_build
cp $HOME/pip_build/dist/mxnet* $HOME/pip_build
