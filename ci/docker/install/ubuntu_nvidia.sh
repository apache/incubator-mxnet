#!/bin/bash

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
apt install -y software-properties-common

# Adding ppas frequently fails due to busy gpg servers, retry 5 times with 5 minute delays.
for i in 1 2 3 4 5; do add-apt-repository -y ppa:graphics-drivers && break || sleep 300; done

# Retrieve ppa:graphics-drivers and install nvidia-drivers.
# Note: DEBIAN_FRONTEND required to skip the interactive setup steps
apt update
DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends cuda-9-1
