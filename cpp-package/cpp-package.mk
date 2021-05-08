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

ifndef LINT_LANG
	LINT_LANG="all"
endif

ifdef CAFFE_PATH
export LD_LIBRARY_PATH=$(CAFFE_PATH)/lib
endif

CPP_PACKAGE_OP_H_FILE = cpp-package/include/mxnet-cpp/op.h

EXTRA_PACKAGES += cpp-package-all
EXTRA_PACKAGES_CLEAN += cpp-package-clean

.PHONY: cpp-package-all cpp-package-lint cpp-package-clean

cpp-package-all: $(CPP_PACKAGE_OP_H_FILE)

cpp-package-clean:
	rm -f $(CPP_PACKAGE_OP_H_FILE)

$(CPP_PACKAGE_OP_H_FILE): lib/libmxnet.so cpp-package/scripts/OpWrapperGenerator.py
	(cd cpp-package/scripts; python OpWrapperGenerator.py $(ROOTDIR)/lib/libmxnet.so)

cpp-package-lint:
	(cd cpp-package; python scripts/lint.py dmlc ${LINT_LANG} include example)

include cpp-package/example/example.mk
include cpp-package/example/inference/inference.mk
