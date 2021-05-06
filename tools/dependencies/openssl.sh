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

# This script builds the static library of openssl that can be used as dependency of mxnet.
set -ex
OPENSSL_VERSION=1.1.1b
if [[ ! -f $DEPS_PATH/lib/libssl.a ]] || [[ ! -f $DEPS_PATH/lib/libcrypto.a ]]; then
    # download and build openssl
    >&2 echo "Building openssl..."
    OPENSSL_VERSION=$(echo $OPENSSL_VERSION | sed 's/\./_/g')
    download \
        https://github.com/openssl/openssl/archive/OpenSSL_${OPENSSL_VERSION}.zip \
        ${DEPS_PATH}/openssl.zip
    unzip -q $DEPS_PATH/openssl.zip -d $DEPS_PATH
    pushd .
    cd $DEPS_PATH/openssl-OpenSSL_$OPENSSL_VERSION
    if [[ $PLATFORM == 'linux' ]]; then
        TARGET=linux-$ARCH
    elif [[ $PLATFORM == 'darwin' ]]; then
        TARGET=darwin64-x86_64-cc
    fi
    ./Configure no-shared no-zlib --prefix=$DEPS_PATH --openssldir=$DEPS_PATH/ssl $TARGET
    $MAKE
    $MAKE install
    popd
fi
