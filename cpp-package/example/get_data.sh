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

set -e

mkdir -p data/mnist_data
cd data/mnist_data

download () {
    local URL=$1
    local GZ_FILE_NAME="${URL##*/}"

    local FILE_NAME="${GZ_FILE_NAME%.*}"
    if [[ -f "${FILE_NAME}" ]]; then
        echo "File ${FILE_NAME} already downloaded."
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
    curl ${CURL_OPTIONS} ${URL} -o ${GZ_FILE_NAME}

    if [[ ! -f "${GZ_FILE_NAME}" ]]; then
        echo "File ${URL} couldn't be downloaded!"
        exit 1
    fi

    gzip -d ${GZ_FILE_NAME}
    (($? != 0)) && exit 1 || return 0
}

# MNIST dataset from: http://yann.lecun.com/exdb/mnist/
FILES=(
    "https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    "https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    "https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    "https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    "http://data.mxnet.io/data/mnist_train.csv.gz")

for FILE in ${FILES[@]}; do
    download ${FILE}
done
