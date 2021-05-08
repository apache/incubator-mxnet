<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

This example shows how to extract features with a pretrained model.

Execute `run.sh` to:
- Download a pretrained model
- Download sample pictures (`dog.jpg` and `cat.jpg`)
- Compile the files
- Execute the featurization on `dog.jpg` and `cat.jpg`


Note:
1. The filename of network parameters may vary, line 67 in `feature_extract.cpp` should be updated accordingly.
2. You need to build MXNet from source to get access to the `lib/libmxnet.so` or point `LD_LIBRARY_PATH` to where it is installed in your system
