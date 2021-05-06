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

# Build on Ubuntu 18.04 LTS for LINUX CPU/GPU
set -ex

apt-get update
apt-get install -y software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test -y
add-apt-repository ppa:openjdk-r/ppa -y # Java lib
apt-get update
apt-get install -y git \
    build-essential \
    ninja-build \
    libssl-dev \
    libcurl4-openssl-dev \
    ccache \
    unzip \
    libtool \
    curl \
    wget \
    sudo \
    gnupg \
    gnupg2 \
    gnupg-agent \
    pandoc \
    python3 \
    python3-pip \
    automake \
    pkg-config \
    openjdk-8-jdk \
    patchelf

# gcc-10 required for -moutline-atomics flag
apt-add-repository "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main"
apt-get update
apt install -y gcc-10 g++-10 gfortran-10
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10 --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-10

curl -o apache-maven-3.3.9-bin.tar.gz -L http://www.eu.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz \
    || curl -o apache-maven-3.3.9-bin.tar.gz -L https://search.maven.org/remotecontent?filepath=org/apache/maven/apache-maven/3.3.9/apache-maven-3.3.9-bin.tar.gz

tar xzf apache-maven-3.3.9-bin.tar.gz
mkdir /usr/local/maven
mv apache-maven-3.3.9/ /usr/local/maven/
update-alternatives --install /usr/bin/mvn mvn /usr/local/maven/apache-maven-3.3.9/bin/mvn 1
update-ca-certificates -f

# the version of the pip shipped with ubuntu may be too lower, install a recent version here
python3 -m pip install --upgrade pip

python3 -m pip install --upgrade --ignore-installed nose cpplint==1.3.0 pylint==2.3.1 nose-timer 'numpy<2.0.0,>1.16.0' 'requests<3,>=2.20.0' scipy boto3

# CMake 3.13.2+ is required
wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar -zxvf cmake-3.16.5.tar.gz
cd cmake-3.16.5
./bootstrap
make -j$(nproc)
sudo make install

