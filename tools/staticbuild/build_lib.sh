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

set -eo pipefail

# This script builds the libraries of mxnet.
cmake_config=${CURDIR}/config/distribution/${PLATFORM}_${VARIANT}.cmake
if [[ ! -f $cmake_config ]]; then
    >&2 echo "Couldn't find cmake config $cmake_config for the current settings."
    exit 1
fi

git submodule update --init --recursive || true

# Build libmxnet.so
rm -rf build; mkdir build; cd build
cmake -GNinja -C $cmake_config \
      -DCMAKE_PREFIX_PATH=${DEPS_PATH} \
      -DCMAKE_FIND_ROOT_PATH=${DEPS_PATH} \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13 \
      ..
ninja
cd -

# Move to lib
rm -rf lib; mkdir lib;
if [[ $PLATFORM == 'linux' ]]; then
    cp -L build/libmxnet.so lib/libmxnet.so
    cp -L $(ldd lib/libmxnet.so | grep libgfortran |  awk '{print $3}') lib/
elif [[ $PLATFORM == 'darwin' ]]; then
    cp -L build/libmxnet.dylib lib/libmxnet.dylib
fi

# Print the linked objects on libmxnet.so
>&2 echo "Checking linked objects on libmxnet.so..."
if [[ ! -z $(command -v readelf) ]]; then
    readelf -d lib/libmxnet.so
    strip --strip-unneeded lib/libmxnet.so
elif [[ ! -z $(command -v otool) ]]; then
    otool -L lib/libmxnet.dylib
    strip -u -r -x lib/libmxnet.dylib
else
    >&2 echo "Not available"
fi

if [[ ! -L deps ]]; then
    ln -s staticdeps deps
fi
