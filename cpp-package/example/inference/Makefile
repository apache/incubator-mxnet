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


CPPEX_SRC = $(wildcard *.cpp)
CPPEX_EXE = $(patsubst %.cpp, %, $(CPPEX_SRC))
OPENCV_CFLAGS=`pkg-config --cflags opencv`
OPENCV_LDFLAGS=`pkg-config --libs opencv`

CXX=g++


CFLAGS=$(COMMFLAGS) -I../../../3rdparty/tvm/nnvm/include -I../../../3rdparty/dmlc-core/include -I ../../include -I ../../../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas
CPPEX_EXTRA_LDFLAGS := -L../../../lib -lmxnet $(OPENCV_LDFLAGS)

all: $(CPPEX_EXE)

debug: CPPEX_CFLAGS += -DDEBUG -g
debug: all


$(CPPEX_EXE):% : %.cpp
	$(CXX) -std=c++0x $(CFLAGS)  $(CPPEX_CFLAGS) -o $@ $(filter %.cpp %.a, $^) $(CPPEX_EXTRA_LDFLAGS)

clean:
	rm -f $(CPPEX_EXE)
