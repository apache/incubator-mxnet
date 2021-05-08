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

CPPEX_SRC = $(wildcard cpp-package/example/*.cpp)
CPPEX_EXE = $(patsubst cpp-package/example/%.cpp, build/cpp-package/example/%, $(CPPEX_SRC))

CPPEX_CFLAGS += -Icpp-package/include
CPPEX_EXTRA_LDFLAGS := -L$(ROOTDIR)/lib -lmxnet

EXTRA_PACKAGES += cpp-package-example-all
EXTRA_PACKAGES_CLEAN += cpp-package-example-clean

.PHONY: cpp-package-example-all cpp-package-example-clean

cpp-package-example-all: cpp-package-all $(CPPEX_EXE)

build/cpp-package/example/% : cpp-package/example/%.cpp lib/libmxnet.so $(CPP_PACKAGE_OP_H_FILE)
	@mkdir -p $(@D)
	$(CXX) -std=c++11 $(CFLAGS) $(CPPEX_CFLAGS) -MM -MT cpp-package/example/$* $< >build/cpp-package/example//$*.d
	$(CXX) -std=c++11 $(CFLAGS) $(CPPEX_CFLAGS) -o $@ $(filter %.cpp %.a, $^) $(LDFLAGS) $(CPPEX_EXTRA_LDFLAGS)

cpp-package-example-clean:
	rm -rf build/cpp-package/example/*

-include build/cpp-package/example/*.d
