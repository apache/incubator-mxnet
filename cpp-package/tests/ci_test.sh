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

set -e # exit on the first error
cd $(dirname $(readlink -f $0))/../example
echo $PWD
export LD_LIBRARY_PATH=$(readlink -f /work/build):$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
ls -l /work/build/

./get_data.sh

cp /work/build/cpp-package/example/lenet .
./lenet 1

cp /work/build/cpp-package/example/alexnet .
./alexnet 1

cp /work/build/cpp-package/example/lenet_with_mxdataiter .
./lenet_with_mxdataiter 1

cp /work/build/cpp-package/example/resnet .
./resnet 1

cp /work/build/cpp-package/example/inception_bn .
./inception_bn 1

cp /work/build/cpp-package/example/mlp .
./mlp 150

cp /work/build/cpp-package/example/mlp_cpu .
./mlp_cpu

cp /work/build/cpp-package/example/mlp_gpu .
./mlp_gpu

cp /work/build/cpp-package/example/test_optimizer .
./test_optimizer

cp /work/build/cpp-package/example/test_kvstore .
./test_kvstore

cp /work/build/cpp-package/example/test_score .
./test_score 0.93

cp /work/build/cpp-package/example/test_ndarray_copy .
./test_ndarray_copy

# skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/20011
#cp ../../build/cpp-package/example/test_regress_label .
#./test_regress_label

# sh unittests/unit_test_mlp_csv.sh

cd inference

cp /work/build/cpp-package/example/sentiment_analysis_rnn .
./unit_test_sentiment_analysis_rnn.sh

cd multi_threaded_inference

cp ../../../../build/cpp-package/example/multi_threaded_inference .
./unit_test_multi_threaded_inference.sh

cd ../..
