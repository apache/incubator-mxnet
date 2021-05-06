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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex
# FIXME(larroy) enable in a different PR
#perl -pi -e 's/archive.ubuntu.com/us-west-2.ec2.archive.ubuntu.com/' /etc/apt/sources.list
apt-get update || true

# Avoid interactive package installers such as tzdata.
export DEBIAN_FRONTEND=noninteractive

apt-get install -y valgrind


VERSION=3.7.10
wget https://www.python.org/ftp/python/$VERSION/Python-$VERSION.tgz
tar -xvzf Python-$VERSION.tgz
cd Python-$VERSION
./configure --with-pydebug --without-pymalloc --with-valgrind --prefix /opt/debugpython/
sudo make OPT=-g && sudo make install

## Add python valgrind suppression file
cp $DIR/Python-$VERSION/Misc/valgrind-python.supp $HOME/workspace/incubator-mxnet
echo "export PATH=/usr/bin:/bin:/opt/debugpython/bin:\$PATH" >> ${HOME}/.bashrc
sudo update-alternatives --install /usr/bin/python python /opt/debugpython/bin/python3 10

rm ../Python-$VERSION.tgz

git clone git://sourceware.org/git/valgrind.git
cd valgrind
./autogen.sh
./configure --prefix=/opt/valgrind
make
sudo make install
echo "export PATH=/opt/valgrind/bin:\$PATH" >> ${HOME}/.bashrc
