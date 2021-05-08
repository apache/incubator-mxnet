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

CXX=g++
BLAS=-L /opt/openblas/lib -lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0
CUDA=-DMSHADOW_USE_CUDA=1
OPENCV_CFLAGS=`pkg-config --cflags opencv`
OPENCV_LDFLAGS=`pkg-config --libs opencv`

CFLAGS=$(COMMFLAGS) -I../../../3rdparty/nnvm/include -I../../../3rdparty/dmlc-core/include -I ../../include -I ../../../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -fopenmp
LDFLAGS=$(COMMFLAGS) -L ../../../lib -lmxnet $(BLAS) $(CUDA) -lgomp -pthread

all: feature_extract prepare_data_with_opencv

feature_extract: ./feature_extract.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

prepare_data_with_opencv: ./prepare_data_with_opencv.cpp
	$(CXX) -c -std=c++11 $(OPENCV_CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(OPENCV_LDFLAGS)
	-rm -f $(basename $@).o

clean:
	-rm -f feature_extract
	-rm -f prepare_data_with_opencv
