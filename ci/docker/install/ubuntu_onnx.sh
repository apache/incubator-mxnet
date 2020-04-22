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

######################################################################
# This script installs ONNX for Python along with all required dependencies 
# on a Ubuntu Machine.
# Tested on Ubuntu 16.04 distro.
######################################################################

set -e
set -x

echo "Installing libprotobuf-dev and protobuf-compiler ..."
apt-get update || true
apt-get install -y libprotobuf-dev protobuf-compiler

echo "Installing pytest, pytest-cov, protobuf, Pillow, ONNX and tabulate ..."
pip3 install pytest==3.6.3 pytest-cov==2.5.1 protobuf==3.5.2 onnx==1.5.0 Pillow==5.0.0 tabulate==0.7.5
