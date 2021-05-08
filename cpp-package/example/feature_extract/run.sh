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

# Downloading the data and model
mkdir -p model
wget -nc -O model/Inception-BN-symbol.json \
    http://data.mxnet.io/mxnet/models/imagenet/inception-bn/Inception-BN-symbol.json
wget -nc -O model/synset.txt \
    http://data.mxnet.io/mxnet/models/imagenet/synset.txt
wget -nc -O model/Inception-BN-0126.params \
    http://data.mxnet.io/mxnet/models/imagenet/inception-bn/Inception-BN-0126.params?raw=true 
wget -nc -O cat.jpg https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true
wget -nc -O dog.jpg https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true
wget -nc -O model/mean_224.nd https://github.com/dmlc/web-data/raw/master/mxnet/example/feature_extract/mean_224.nd
tar -xvzf inception-bn.tar.gz -C model --skip-old-files

# Building
make

# Preparing the data
./prepare_data_with_opencv

# Running the featurization
LD_LIBRARY_PATH=../../../lib ./feature_extract
