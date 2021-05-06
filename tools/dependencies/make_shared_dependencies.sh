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

# This is a convenience script for calling the build scripts of all dependency libraries.
# Environment variables should be set beforehand.

set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

download () {
    local URL=$1
    local OUT_FILE=$2

    if [[ -f "${OUT_FILE}" ]]; then
        echo "File ${OUT_FILE} already downloaded."
        return 0
    fi

    echo "Downloading ${URL} ..."
    local CURL_OPTIONS="--connect-timeout 10 \
              --max-time 300 \
              --retry-delay 10 \
              --retry 3 \
              --retry-delay 0 \
              --location \
              --silent"
    curl ${CURL_OPTIONS} ${URL} -o ${OUT_FILE}

    if [[ ! -f "${OUT_FILE}" ]]; then
        echo "File ${URL} couldn't be downloaded!"
        exit 1
    fi
}

if [[ ! $PLATFORM == 'darwin' ]]; then
    source ${DIR}/openblas.sh
fi
source $DIR/libz.sh
source $DIR/libturbojpeg.sh
source $DIR/libpng.sh
if [[ ! $ARCH == 'aarch64' ]]; then
    source $DIR/libtiff.sh
fi
source $DIR/openssl.sh
source $DIR/curl.sh
source $DIR/eigen.sh
source $DIR/opencv.sh
if [[ ! $ARCH == 'aarch64' ]]; then
    source $DIR/protobuf.sh
    source $DIR/cityhash.sh
fi
source $DIR/zmq.sh
source $DIR/lz4.sh

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(dirname $(find $DEPS_PATH -type f -name 'libprotoc*' | grep protobuf | head -n 1)):$DEPS_PATH/lib
