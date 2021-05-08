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

# This file is a unit test for mlp_csv.cpp example in 'example' directory.
# The file
#    1. Downloads the MNIST data,
#    2. Converts it into CSV format.
#    3. Runs the mlp_csv example and ensures that the accuracy is more than expected.
#

#!/bin/bash

set -e # exit on the first error
export EXE_NAME=mlp_csv

cd $(dirname $(readlink -f $0))/../
export LD_LIBRARY_PATH=$(readlink -f ../../build):$LD_LIBRARY_PATH

if [ ! -f ../../build/cpp-package/example/${EXE_NAME} ];
then
echo "FAIL: ${EXE_NAME} does not exist"
exit
fi

cp ../../build/cpp-package/example/${EXE_NAME} .

./get_data.sh
python3 mnist_to_csv.py ./data/mnist_data/train-images-idx3-ubyte ./data/mnist_data/train-labels-idx1-ubyte ./data/mnist_data/mnist_train.csv 60000
python3 mnist_to_csv.py ./data/mnist_data/t10k-images-idx3-ubyte ./data/mnist_data/t10k-labels-idx1-ubyte ./data/mnist_data/mnist_test.csv 10000

./${EXE_NAME} --train ./data/mnist_data/mnist_train.csv --test ./data/mnist_data/mnist_test.csv --epochs 10 --batch_size 100 --hidden_units "128 64 10" 2&> ${EXE_NAME}.log

if [ ! -f ${EXE_NAME}.log ];
then
echo "FAIL: Log file ${EXE_NAME}.log does not exist."
exit
fi

# Obtain the accuracy achieved by mlp model after training with MNIST data in CSV format.
export Acc_obtained=`grep -oP '.*\K(?<=Accuracy: ).*$' ${EXE_NAME}.log | tail -1 | tr -d '\n'`
export Acc_expected=0.98

# If the obtained accuracy does not meet the expected accuracy, report the test as FAIL.
if [ $(echo "$Acc_obtained $Acc_expected" | awk '{printf($1 >= $2) ? 1 : 0}') -eq 1 ] ;
then
echo "PASS: ${EXE_NAME} obtained $Acc_obtained accuracy."
else
echo "FAIL: Accuracy = $Acc_obtained is less than expected accuracy $Acc_expected."
fi
