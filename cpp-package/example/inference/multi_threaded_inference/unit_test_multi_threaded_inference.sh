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

# http://mxnet.apache.org/versions/master/api/cpp/docs/tutorials/multi_threaded_inference.html

# Install test data.
wget https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/data/grace_hopper.jpg
wget http://optipng.sourceforge.net/pngtech/img/lena.png

# Get Model.
python3 get_model.py --model imagenet1k-inception-bn

# Run test
./multi_threaded_inference imagenet1k-inception-bn 1 grace_hopper.jpg lena.png
