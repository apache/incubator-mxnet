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

set -ex
# create ./model directory if not existed
if [ ! -d model ]; then
    mkdir -p model
fi
# create ./data directory if not existed
if [ ! -d data ]; then
    mkdir -p data
fi
# Downloading the data and model if not existed
model_file=./model/Inception-BN-symbol.json
params_file=./model/Inception-BN-0126.params
if [ ! -f ${model_file} ] || [ ! -f ${params_file} ]; then
    wget -nc http://data.mxnet.io/models/imagenet/inception-bn.tar.gz
    tar -xvzf inception-bn.tar.gz -C model
fi
cd model
wget -nc https://raw.githubusercontent.com/dmlc/gluon-cv/master/gluoncv/model_zoo/quantized/resnet50_v1_int8-symbol.json
cd ../data
wget -nc http://data.mxnet.io/data/val_256_q90.rec
cd ..

# Running inference on imagenet.
if [ "$(uname)" == "Darwin" ]; then
    echo ">>> INFO: FP32 real data"
    DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:../../../lib ./imagenet_inference --symbol_file "./model/Inception-BN-symbol.json" --params_file "./model/Inception-BN-0126.params" --dataset "./data/val_256_q90.rec" --rgb_mean "123.68 116.779 103.939" --batch_size 1 --num_skipped_batches 50 --num_inference_batches 500

    echo ">>> INFO: FP32 dummy data"
    DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:../../../lib ./imagenet_inference --symbol_file "./model/Inception-BN-symbol.json" --batch_size 1 --num_inference_batches 500 --benchmark
else
    echo ">>> INFO: FP32 real data"
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../../lib ./imagenet_inference --symbol_file "./model/Inception-BN-symbol.json" --params_file "./model/Inception-BN-0126.params" --dataset "./data/val_256_q90.rec" --rgb_mean "123.68 116.779 103.939" --batch_size 1 --num_skipped_batches 50 --num_inference_batches 500

    echo ">>> INFO: FP32 dummy data"
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../../lib ./imagenet_inference --symbol_file "./model/Inception-BN-symbol.json" --batch_size 1 --num_inference_batches 500 --benchmark

    lib_name=$(ls -a ../../../lib | grep -oE 'mkldnn' | tail -1)
    if [[ -n ${lib_name} ]] && [[ 'mkldnn' =~ ${lib_name} ]]; then
        echo ">>> INFO: INT8 dummy data"
        LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../../lib ./imagenet_inference --symbol_file "./model/resnet50_v1_int8-symbol.json" --batch_size 1 --num_inference_batches 500 --benchmark
    else
        echo "Skipped INT8 test because mkldnn was not found which is required for running inference with quantized models."
    fi
fi
