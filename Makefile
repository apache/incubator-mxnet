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

ROOTDIR = $(CURDIR)
TPARTYDIR = $(ROOTDIR)/3rdparty

ifeq ($(OS),Windows_NT)
	UNAME_S := Windows
else
	UNAME_S := $(shell uname -s)
	UNAME_P := $(shell uname -p)
endif

ifndef config
ifdef CXXNET_CONFIG
	config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif
endif

ifndef DMLC_CORE
	DMLC_CORE = $(TPARTYDIR)/dmlc-core
endif
CORE_INC = $(wildcard $(DMLC_CORE)/include/*/*.h)

ifndef NNVM_PATH
	NNVM_PATH = $(TPARTYDIR)/tvm/nnvm
endif

ifndef DLPACK_PATH
	DLPACK_PATH = $(ROOTDIR)/3rdparty/dlpack
endif

ifndef AMALGAMATION_PATH
	AMALGAMATION_PATH = $(ROOTDIR)/amalgamation
endif

ifndef TVM_PATH
	TVM_PATH = $(TPARTYDIR)/tvm
endif

ifndef LLVM_PATH
	LLVM_PATH = $(TVM_PATH)/build/llvm
endif

ifneq ($(USE_OPENMP), 1)
	export NO_OPENMP = 1
endif

# use customized config file
include $(config)

ifndef USE_MKLDNN
ifneq ($(UNAME_S), Darwin)
ifneq ($(UNAME_S), Windows)
ifeq ($(UNAME_P), x86_64)
	USE_MKLDNN=1
endif
endif
endif
endif

ifeq ($(USE_MKL2017), 1)
$(warning "USE_MKL2017 is deprecated. We will switch to USE_MKLDNN.")
	USE_MKLDNN=1
endif

ifeq ($(USE_MKLDNN), 1)
	MKLDNNROOT = $(ROOTDIR)/3rdparty/mkldnn/build/install
endif

ifndef USE_INTGEMM
ifeq ($(UNAME_P), x86_64)
	COMPILER := $(shell $(CXX) --version |head -n 1 |cut -d " " -f 1)
	COMPILER_VERSION := $(shell $(CXX) -dumpversion |cut -d . -f 1)
	ifeq ($(COMPILER), clang)
		USE_INTGEMM=1
	endif
	ifeq ($(COMPILER), Apple)
		USE_INTGEMM=1
	endif
	# If it's not clang and not Apple clang, it's probably gcc and we need at least 5.
	# gcc --version gives the name of the program it was called with, which makes it hard to detect.
	COMPILER_VERSION_GE_5 := $(shell expr $(COMPILER_VERSION) \>= 5)
	ifeq ($(COMPILER_VERSION_GE_5), 1)
		USE_INTGEMM=1
	endif
endif
endif

include $(TPARTYDIR)/mshadow/make/mshadow.mk
include $(DMLC_CORE)/make/dmlc.mk

# all tge possible warning tread
WARNFLAGS= -Wall -Wsign-compare
CFLAGS = -DMSHADOW_FORCE_STREAM $(WARNFLAGS)
# use old thread local implementation in DMLC-CORE
CFLAGS += -DDMLC_MODERN_THREAD_LOCAL=0
# disable stack trace in exception by default.
CFLAGS += -DDMLC_LOG_STACK_TRACE_SIZE=0
CFLAGS += -DDMLC_LOG_FATAL_THROW=1

ifeq ($(DEV), 1)
  # Excluded from Werror:
  # 1) variables used in '#pragma omp parallel' are considered unused
	CFLAGS += -g -Werror -Wno-error=unused-variable -Wno-error=maybe-uninitialized -Wno-error=unused-function
	NVCCFLAGS += -Werror cross-execution-space-call
endif

# CFLAGS for debug
ifeq ($(DEBUG), 1)
	CFLAGS += -g -O0 -D_GLIBCXX_ASSERTIONS
else
	CFLAGS += -O3 -DNDEBUG=1
endif
CFLAGS += -I$(TPARTYDIR)/mshadow/ -I$(TPARTYDIR)/dmlc-core/include -fPIC -I$(NNVM_PATH)/include -I$(DLPACK_PATH)/include -I$(TPARTYDIR)/tvm/include -Iinclude $(MSHADOW_CFLAGS)
LDFLAGS = -pthread -ldl $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)

# please note that when you enable this, you might run into an linker not being able to work properly due to large code injection.
# you can find more information here https://github.com/apache/incubator-mxnet/issues/15971
ifeq ($(ENABLE_TESTCOVERAGE), 1)
        CFLAGS += --coverage
        LDFLAGS += --coverage
endif

ifeq ($(USE_NVTX), 1)
        CFLAGS += -DMXNET_USE_NVTX=1
        LDFLAGS += -lnvToolsExt
endif

ifeq ($(USE_TENSORRT), 1)
	CFLAGS +=  -I$(ROOTDIR) -I$(TPARTYDIR) -DONNX_NAMESPACE=$(ONNX_NAMESPACE) -DMXNET_USE_TENSORRT=1
	LDFLAGS += -lprotobuf -pthread -lonnx -lonnx_proto -lnvonnxparser -lnvonnxparser_runtime -lnvinfer -lnvinfer_plugin
endif
# -L/usr/local/lib

ifeq ($(DEBUG), 1)
	NVCCFLAGS += -std=c++11 -Xcompiler -D_FORCE_INLINES -g -G -O0 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
else
	NVCCFLAGS += -std=c++11 -Xcompiler -D_FORCE_INLINES -O3 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
endif

# CFLAGS for segfault logger
ifeq ($(USE_SIGNAL_HANDLER), 1)
	CFLAGS += -DMXNET_USE_SIGNAL_HANDLER=1
endif

# Caffe Plugin
ifdef CAFFE_PATH
	CFLAGS += -DMXNET_USE_CAFFE=1
endif

ifndef LINT_LANG
	LINT_LANG = "all"
endif

ifeq ($(USE_MKLDNN), 1)
	CFLAGS += -DMXNET_USE_MKLDNN=1
	CFLAGS += -I$(ROOTDIR)/src/operator/nn/mkldnn/
	CFLAGS += -I$(MKLDNNROOT)/include
	LIB_DEP += $(MKLDNNROOT)/lib/libdnnl.a
endif

# Use MKL's layernorm implementation.  Only has an impact if MKL is compiled in.
ifeq ($(USE_MKL_LAYERNORM), 1)
  CFLAGS += -DMXNET_USE_MKL_LAYERNORM=1
endif

# setup opencv
ifeq ($(USE_OPENCV), 1)
	CFLAGS += -DMXNET_USE_OPENCV=1
	ifneq ($(filter-out NONE, $(USE_OPENCV_INC_PATH)),)
                ifneq ($(wildcard $(USE_OPENCV_INC_PATH)/include/opencv4/opencv2/opencv*),)
                    opencv_inc = -I$(USE_OPENCV_INC_PATH)/include/opencv4
                endif
                ifneq ($(wildcard $(USE_OPENCV_INC_PATH)/include/opencv2/opencv*),)
                    opencv_inc = -I$(USE_OPENCV_INC_PATH)/include
                endif
                ifneq ($(wildcard $(USE_OPENCV_INC_PATH)/opencv4/opencv2/opencv*),)
                    opencv_inc = -I$(USE_OPENCV_INC_PATH)/opencv4
                endif
                ifneq ($(wildcard $(USE_OPENCV_INC_PATH)/opencv2/opencv*),)
                    opencv_inc = -I$(USE_OPENCV_INC_PATH)
                endif
                ifneq ($(filter-out NONE, $(opencv_inc)),)
                    CFLAGS += $(opencv_inc)
                else
$(error Cannot determine OpenCV include path)
                endif
		ifeq ($(filter-out NONE, $(USE_OPENCV_LIB_PATH)),)
$(error Please add the path of OpenCV shared library path into `USE_OPENCV_LIB_PATH`, when `USE_OPENCV_INC_PATH` is not NONE)
		endif
		LDFLAGS += -L$(USE_OPENCV_LIB_PATH)
		ifneq ($(wildcard $(USE_OPENCV_LIB_PATH)/libopencv_imgcodecs.*),)
			LDFLAGS += -lopencv_imgcodecs
		endif
		ifneq ($(wildcard $(USE_OPENCV_LIB_PATH)/libopencv_highgui.*),)
			LDFLAGS += -lopencv_highgui
		endif
	else
		ifeq ("$(shell pkg-config --exists opencv4; echo $$?)", "0")
			OPENCV_LIB = opencv4
		else
			OPENCV_LIB = opencv
		endif
		CFLAGS += $(shell pkg-config --cflags $(OPENCV_LIB))
		LDFLAGS += $(shell pkg-config --libs-only-L $(OPENCV_LIB))
		LDFLAGS += $(filter -lopencv_imgcodecs -lopencv_highgui, $(shell pkg-config --libs-only-l $(OPENCV_LIB)))
	endif
	LDFLAGS += -lopencv_imgproc -lopencv_core
	BIN += bin/im2rec
else
	CFLAGS += -DMXNET_USE_OPENCV=0
endif

ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
	CFLAGS += -DMXNET_USE_OPENMP=1
endif

ifeq ($(USE_NNPACK), 1)
	CFLAGS += -DMXNET_USE_NNPACK=1
	LDFLAGS += -lnnpack
endif

ifeq ($(USE_OPERATOR_TUNING), 1)
	CFLAGS += -DMXNET_USE_OPERATOR_TUNING=1
endif

ifeq ($(USE_INT64_TENSOR_SIZE), 1)
   CFLAGS += -DMSHADOW_INT64_TENSOR_SIZE=1
else
   CFLAGS += -DMSHADOW_INT64_TENSOR_SIZE=0
endif
# verify existence of separate lapack library when using blas/openblas/atlas
# switch off lapack support in case it can't be found
# issue covered with this
#   -  for Ubuntu 14.04 or lower, lapack is not automatically installed with openblas
#   -  for Ubuntu, installing atlas will not automatically install the atlas provided lapack library
#   -  for rhel7.2, try installing the package `lapack-static` via yum will dismiss this warning.
# silently switching lapack off instead of letting the build fail because of backward compatibility
ifeq ($(USE_LAPACK), 1)
ifeq ($(USE_BLAS),$(filter $(USE_BLAS),blas openblas atlas mkl))
ifeq (,$(wildcard $(USE_LAPACK_PATH)/liblapack.a))
ifeq (,$(wildcard $(USE_LAPACK_PATH)/liblapack.so))
ifeq (,$(wildcard $(USE_LAPACK_PATH)/liblapack.dylib))
ifeq (,$(wildcard /lib/liblapack.a))
ifeq (,$(wildcard /lib/liblapack.so))
ifeq (,$(wildcard /usr/lib/liblapack.a))
ifeq (,$(wildcard /usr/lib/liblapack.so))
ifeq (,$(wildcard /usr/lib/liblapack.dylib))
ifeq (,$(wildcard /usr/lib64/liblapack.a))
ifeq (,$(wildcard /usr/lib64/liblapack.so))
	USE_LAPACK = 0
        $(warning "USE_LAPACK disabled because libraries were not found")
endif
endif
endif
endif
endif
endif
endif
endif
endif
endif
endif
endif

# lapack settings.
ifeq ($(USE_LAPACK), 1)
	ifneq ($(USE_LAPACK_PATH), )
		LDFLAGS += -L$(USE_LAPACK_PATH)
	endif
	ifeq ($(USE_BLAS),$(filter $(USE_BLAS),blas openblas atlas mkl))
		LDFLAGS += -llapack
	endif
	CFLAGS += -DMXNET_USE_LAPACK
endif

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DMSHADOW_USE_CUDNN=1
	LDFLAGS += -lcudnn
endif

ifeq ($(USE_BLAS), openblas)
	CFLAGS += -DMXNET_USE_BLAS_OPEN=1
else ifeq ($(USE_BLAS), atlas)
	CFLAGS += -DMXNET_USE_BLAS_ATLAS=1
else ifeq ($(USE_BLAS), mkl)
	CFLAGS += -DMXNET_USE_BLAS_MKL=1
else ifeq ($(USE_BLAS), apple)
	CFLAGS += -DMXNET_USE_BLAS_APPLE=1
endif

# whether to use F16C instruction set extension for fast fp16 compute on CPU
# if cross compiling you may want to explicitly turn it off if target system does not support it
ifndef USE_F16C
    ifneq ($(OS),Windows_NT)
        detected_OS := $(shell uname -s)
        ifeq ($(detected_OS),Darwin)
            F16C_SUPP = $(shell sysctl -a | grep machdep.cpu.features | grep F16C)
        endif
        ifeq ($(detected_OS),Linux)
            F16C_SUPP = $(shell cat /proc/cpuinfo | grep flags | grep f16c)
        endif
	ifneq ($(strip $(F16C_SUPP)),)
                USE_F16C=1
        else
                USE_F16C=0
        endif
    endif
    # if OS is Windows, check if your processor and compiler support F16C architecture.
    # One way to check if processor supports it is to download the tool
    # https://docs.microsoft.com/en-us/sysinternals/downloads/coreinfo.
    # If coreinfo -c shows F16C and compiler supports it,
    # then you can set USE_F16C=1 explicitly to leverage that capability"
endif

# gperftools malloc library (tcmalloc)
ifeq ($(USE_GPERFTOOLS), 1)
FIND_LIBFILEEXT=so
ifeq ($(USE_GPERFTOOLS_STATIC), 1)
FIND_LIBFILEEXT=a
endif
FIND_LIBFILE=$(wildcard $(USE_GPERFTOOLS_PATH)/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /lib/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/local/lib/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib64/libtcmalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
	USE_GPERFTOOLS=0
endif
endif
endif
endif
endif
ifeq ($(USE_GPERFTOOLS), 1)
	CFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
	LDFLAGS += $(FIND_LIBFILE)
endif

# jemalloc malloc library (if not using gperftools)
else
ifeq ($(USE_JEMALLOC), 1)
FIND_LIBFILEEXT=so
ifeq ($(USE_JEMALLOC_STATIC), 1)
FIND_LIBFILEEXT=a
endif
FIND_LIBFILE=$(wildcard $(USE_JEMALLOC_PATH)/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /lib/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/local/lib/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib/x86_64-linux-gnu/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
FIND_LIBFILE=$(wildcard /usr/lib64/libjemalloc.$(FIND_LIBFILEEXT))
ifeq (,$(FIND_LIBFILE))
	USE_JEMALLOC=0
endif
endif
endif
endif
endif
endif
ifeq ($(USE_JEMALLOC), 1)
	CFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc \
	-fno-builtin-free -DUSE_JEMALLOC
	LDFLAGS += $(FIND_LIBFILE)
endif
endif
endif

# If not using tcmalloc or jemalloc, print a warning (user should consider installing)
ifneq ($(USE_GPERFTOOLS), 1)
ifneq ($(USE_JEMALLOC), 1)
$(warning WARNING: Significant performance increases can be achieved by installing and \
enabling gperftools or jemalloc development packages)
endif
endif

ifeq ($(USE_THREADED_ENGINE), 1)
	CFLAGS += -DMXNET_USE_THREADED_ENGINE
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

ifeq ($(NVCC), NONE)
	# If NVCC has not been manually defined, use the CUDA_PATH bin dir.
	ifneq ($(USE_CUDA_PATH), NONE)
		NVCC=$(USE_CUDA_PATH)/bin/nvcc
	endif
endif

# Guard against displaying nvcc info messages to users not using CUDA.
ifeq ($(USE_CUDA), 1)
	# Get AR version, compare with expected ar version and find bigger and smaller version of the two
	AR_VERSION := $(shell ar --version | egrep -o "([0-9]{1,}\.)+[0-9]{1,}")
	EXPECTED_AR_VERSION := $(shell echo "2.27")
	LARGE_VERSION := $(shell printf '%s\n' "$(AR_VERSION)" "$(EXPECTED_AR_VERSION)" | sort -V | tail -n 1)
	SMALL_VERSION := $(shell printf '%s\n' "$(AR_VERSION)" "$(EXPECTED_AR_VERSION)" | sort -V | head -n 1)

	# If NVCC is not at the location specified, use CUDA_PATH instead.
	ifeq ("$(wildcard $(NVCC))","")
		ifneq ($(USE_CUDA_PATH), NONE)
			NVCC=$(USE_CUDA_PATH)/bin/nvcc

# if larger version is the expected one and larger != smaller
# this means ar version is less than expected version and user needs to be warned
ifeq ($(LARGE_VERSION), $(EXPECTED_AR_VERSION))
ifneq ($(LARGE_VERSION), $(SMALL_VERSION))
define n


endef

$(warning WARNING: Archive utility: ar version being used is less than 2.27.0. $n \
		   Note that with USE_CUDA=1 flag and USE_CUDNN=1 this is known to cause problems. $n \
		   For more info see: https://github.com/apache/incubator-mxnet/issues/15084)
$(shell sleep 5)
endif
endif
$(info INFO: nvcc was not found on your path)
$(info INFO: Using $(NVCC) as nvcc path)
		else
$(warning WARNING: could not find nvcc compiler, the specified path was: $(NVCC))
		endif
	endif
endif

# Sets 'CUDA_ARCH', which determines the GPU architectures supported
# by the compiled kernels.  Users can edit the KNOWN_CUDA_ARCHS list below
# to remove archs they don't wish to support to speed compilation, or they can
# pre-set the CUDA_ARCH args in config.mk to a non-null value for full control.
#
# For archs in this list, nvcc will create a fat-binary that will include
# the binaries (SASS) for all architectures supported by the installed version
# of the cuda toolkit, plus the assembly (PTX) for the most recent such architecture.
# If these kernels are then run on a newer-architecture GPU, the binary will
# be JIT-compiled by the updated driver from the included PTX.
ifeq ($(USE_CUDA), 1)
ifeq ($(CUDA_ARCH),)
	KNOWN_CUDA_ARCHS := 30 35 50 52 60 61 70 75 80
	# Run nvcc on a zero-length file to check architecture-level support.
	# Create args to include SASS in the fat binary for supported levels.
	CUDA_ARCH := $(foreach arch,$(KNOWN_CUDA_ARCHS), \
				$(shell $(NVCC) -arch=sm_$(arch) -E --x cu /dev/null >/dev/null 2>&1 && \
						echo -gencode arch=compute_$(arch),code=sm_$(arch)))
	# Convert a trailing "code=sm_NN" to "code=[sm_NN,compute_NN]" to also
	# include the PTX of the most recent arch in the fat-binaries for
	# forward compatibility with newer GPUs.
	CUDA_ARCH := $(shell echo $(CUDA_ARCH) | sed 's/sm_\([0-9]*\)$$/[sm_\1,compute_\1]/')
	# Add fat binary compression if supported by nvcc.
	COMPRESS := --fatbin-options -compress-all
	CUDA_ARCH += $(shell $(NVCC) -cuda $(COMPRESS) --x cu /dev/null -o /dev/null >/dev/null 2>&1 && \
						 echo $(COMPRESS))
endif
$(info Running CUDA_ARCH: $(CUDA_ARCH))
endif

# ps-lite
PS_PATH=$(ROOTDIR)/3rdparty/ps-lite
DEPS_PATH=$(shell pwd)/deps
include $(PS_PATH)/make/ps.mk
ifeq ($(USE_DIST_KVSTORE), 1)
	CFLAGS += -DMXNET_USE_DIST_KVSTORE -I$(PS_PATH)/include -I$(DEPS_PATH)/include
	LIB_DEP += $(PS_PATH)/build/libps.a
	LDFLAGS += $(PS_LDFLAGS_A)
endif

.PHONY: clean all extra-packages test lint clean_all rcpplint rcppexport roxygen\
	cython3 cython cyclean

all: lib/libmxnet.a lib/libmxnet.so $(BIN) extra-packages extension_libs

SRC = $(wildcard src/*/*/*/*.cc src/*/*/*.cc src/*/*.cc src/*.cc)

ifeq ($(USE_INTGEMM), 1)
	ifndef INTGEMM_PATH
		INTGEMM_PATH = build/3rdparty/intgemm
	endif
	CFLAGS += -DMXNET_USE_INTGEMM=1
	LIB_DEP += $(INTGEMM_PATH)/libintgemm.a

# Download intgemm if it isn't already
$(INTGEMM_PATH):
	@mkdir -p $(INTGEMM_PATH)
	rm -rf $(INTGEMM_PATH)
	git clone https://github.com/kpu/intgemm $(INTGEMM_PATH)
	cd $(INTGEMM_PATH) && git checkout -q 4172dcc209e6793dd920dec9cf9c9fc81605bd9d

$(INTGEMM_PATH)/compile_test_avx512bw.cc: $(INTGEMM_PATH)
	@
$(INTGEMM_PATH)/compile_test_avx512vnni.cc: $(INTGEMM_PATH)
	@
$(INTGEMM_PATH)/intgemm/intgemm.cc: $(INTGEMM_PATH)
	@

# Compiler tests for AVX512BW and AVX512VNNI.
$(INTGEMM_PATH)/intgemm/intgemm_config.h: $(INTGEMM_PATH)/compile_test_avx512bw.cc $(INTGEMM_PATH)/compile_test_avx512vnni.cc
	echo '#pragma once' >$(INTGEMM_PATH)/intgemm/intgemm_config.h
	$(CXX) $(CFLAGS) $(INTGEMM_PATH)/compile_test_avx512bw.cc 2>/dev/null && echo \#define INTGEMM_COMPILER_SUPPORTS_AVX512BW >>$(INTGEMM_PATH)/intgemm/intgemm_config.h || echo Your compiler is missing AVX512BW support
	$(CXX) $(CFLAGS) $(INTGEMM_PATH)/compile_test_avx512vnni.cc 2>/dev/null && echo \#define INTGEMM_COMPILER_SUPPORTS_AVX512VNNI >>$(INTGEMM_PATH)/intgemm/intgemm_config.h || echo Your compiler is missing AVX512VNNI support

$(INTGEMM_PATH)/intgemm/intgemm.o: $(INTGEMM_PATH)/intgemm/intgemm_config.h $(INTGEMM_PATH)/intgemm/intgemm.cc $(wildcard $(INTGEMM_PATH)/intgemm/*.h $(INTGEMM_PATH)/intgemm/*/*.h)
	$(CXX) $(CFLAGS) -I$(INTGEMM_PATH) -std=c++11 -c $(INTGEMM_PATH)/intgemm/intgemm.cc -o $@

$(INTGEMM_PATH)/libintgemm.a: $(INTGEMM_PATH)/intgemm/intgemm.o
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)
else
	#If we're not using intgemm, remove the operators from src.
	INTGEMM_OPS := $(wildcard src/operator/contrib/intgemm/*.cc)
	SRC := $(filter-out $(INTGEMM_OPS),$(SRC))
endif

OBJ = $(patsubst %.cc, build/%.o, $(SRC))
CUSRC = $(wildcard src/*/*/*/*.cu src/*/*/*.cu src/*/*.cu src/*.cu)
CUOBJ = $(patsubst %.cu, build/%_gpu.o, $(CUSRC))

ifeq ($(USE_TVM_OP), 1)
LIB_DEP += lib/libtvm_runtime.so lib/libtvmop.so
CFLAGS += -I$(TVM_PATH)/include -DMXNET_USE_TVM_OP=1
LDFLAGS += -L$(ROOTDIR)/lib -ltvm_runtime -Wl,-rpath,'$${ORIGIN}'

TVM_USE_CUDA := OFF
ifeq ($(USE_CUDA), 1)
	TVM_USE_CUDA := ON
	ifneq ($(USE_CUDA_PATH), NONE)
		TVM_USE_CUDA := $(USE_CUDA_PATH)
	endif
endif
endif

# extra operators
ifneq ($(EXTRA_OPERATORS),)
	EXTRA_SRC = $(wildcard $(patsubst %, %/*.cc, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.cc, $(EXTRA_OPERATORS)))
	EXTRA_OBJ = $(patsubst %.cc, %.o, $(EXTRA_SRC))
	EXTRA_CUSRC = $(wildcard $(patsubst %, %/*.cu, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.cu, $(EXTRA_OPERATORS)))
	EXTRA_CUOBJ = $(patsubst %.cu, %_gpu.o, $(EXTRA_CUSRC))
else
	EXTRA_SRC =
	EXTRA_OBJ =
	EXTRA_CUSRC =
	EXTRA_CUOBJ =
endif

# plugin
PLUGIN_OBJ =
PLUGIN_CUOBJ =
include $(MXNET_PLUGINS)

ifneq ($(UNAME_S), Windows)
	ifeq ($(UNAME_S), Darwin)
		WHOLE_ARCH= -all_load
		NO_WHOLE_ARCH= -noall_load
	else
		WHOLE_ARCH= --whole-archive
		NO_WHOLE_ARCH= --no-whole-archive
	endif
endif

# all dep
LIB_DEP += $(DMLC_CORE)/libdmlc.a $(NNVM_PATH)/lib/libnnvm.a
ALL_DEP = $(OBJ) $(EXTRA_OBJ) $(PLUGIN_OBJ) $(LIB_DEP)

ifeq ($(USE_CUDA), 1)
	CUDA_VERSION_MAJOR := $(shell $(NVCC) --version | grep "release" | awk '{print $$6}' | cut -c2- | cut -d '.' -f1)
	ifeq ($(shell test $(CUDA_VERSION_MAJOR) -lt 11; echo $$?), 0)
		CFLAGS += -I$(ROOTDIR)/3rdparty/nvidia_cub -DCUB_IGNORE_DEPRECATED_CPP_DIALECT
	endif

	ALL_DEP += $(CUOBJ) $(EXTRA_CUOBJ) $(PLUGIN_CUOBJ)
	LDFLAGS += -lcufft
	ifeq ($(ENABLE_CUDA_RTC), 1)
		LDFLAGS += -lcuda -lnvrtc
		CFLAGS += -DMXNET_ENABLE_CUDA_RTC=1
	endif
	# Make sure to add stubs as fallback in order to be able to build
	# without full CUDA install (especially if run without nvidia-docker)
	LDFLAGS += -L/usr/local/cuda/lib64/stubs
	ifeq ($(USE_NCCL), 1)
		ifneq ($(USE_NCCL_PATH), NONE)
			CFLAGS += -I$(USE_NCCL_PATH)/include
			LDFLAGS += -L$(USE_NCCL_PATH)/lib
		endif
		LDFLAGS += -lnccl
		CFLAGS += -DMXNET_USE_NCCL=1
	else
		CFLAGS += -DMXNET_USE_NCCL=0
	endif
else
	CFLAGS += -DMXNET_USE_NCCL=0
endif

ifeq ($(USE_LIBJPEG_TURBO), 1)
	ifneq ($(USE_LIBJPEG_TURBO_PATH), NONE)
		CFLAGS += -I$(USE_LIBJPEG_TURBO_PATH)/include
		LDFLAGS += -L$(USE_LIBJPEG_TURBO_PATH)/lib
	endif
	LDFLAGS += -lturbojpeg
	CFLAGS += -DMXNET_USE_LIBJPEG_TURBO=1
else
	CFLAGS += -DMXNET_USE_LIBJPEG_TURBO=0
endif

ifeq ($(CI), 1)
	MAVEN_ARGS := -B
endif

# For quick compile test, used smaller subset
ALLX_DEP= $(ALL_DEP)

ifeq ($(USE_INTGEMM), 1)
# Enforce a dependency on $(INTGEMM_PATH)/intgemm/intgemm_config.h which is a generated header based on compiler support.
build/src/operator/contrib/intgemm/%.o: src/operator/contrib/intgemm/%.cc $(INTGEMM_PATH)/intgemm/intgemm_config.h | mkldnn
	@mkdir -p $(@D)
	$(CXX) -std=c++11 -c $(CFLAGS) -MMD -I$(INTGEMM_PATH) -Isrc/operator -c $< -o $@
endif

build/src/%.o: src/%.cc | mkldnn
	@mkdir -p $(@D)
	$(CXX) -std=c++11 -c $(CFLAGS) -MMD -c $< -o $@

build/src/%_gpu.o: src/%.cu | mkldnn
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS)" --generate-dependencies -MT build/src/$*_gpu.o $< >build/src/$*_gpu.d
	$(NVCC) -c -o $@ $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS)" $<

# A nvcc bug cause it to generate "generic/xxx.h" dependencies from torch headers.
# Use CXX to generate dependency instead.
build/plugin/%_gpu.o: plugin/%.cu
	@mkdir -p $(@D)
	$(CXX) -std=c++11 $(CFLAGS) -MM -MT build/plugin/$*_gpu.o $< >build/plugin/$*_gpu.d
	$(NVCC) -c -o $@ $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS)" $<

build/plugin/%.o: plugin/%.cc | mkldnn
	@mkdir -p $(@D)
	$(CXX) -std=c++11 -c $(CFLAGS) -MMD -c $< -o $@

%_gpu.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS) -Isrc/operator" --generate-dependencies -MT $*_gpu.o $< >$*_gpu.d
	$(NVCC) -c -o $@ $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS) -Isrc/operator" $<

%.o: %.cc $(CORE_INC)
	@mkdir -p $(@D)
	$(CXX) -std=c++11 -c $(CFLAGS) -MMD -Isrc/operator -c $< -o $@

# Set install path for libmxnet.so on Mac OS
ifeq ($(UNAME_S), Darwin)
        LDFLAGS += -Wl,-install_name,@rpath/libmxnet.so
endif

# NOTE: to statically link libmxnet.a we need the option
# --Wl,--whole-archive -lmxnet --Wl,--no-whole-archive
lib/libmxnet.a: $(ALLX_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libmxnet.so: $(ALLX_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter-out %libnnvm.a, $(filter %.o %.a, $^)) $(LDFLAGS) \
	-Wl,${WHOLE_ARCH} $(filter %libnnvm.a, $^) -Wl,${NO_WHOLE_ARCH}

$(PS_PATH)/build/libps.a: PSLITE

PSLITE:
	$(MAKE) CXX="$(CXX)" DEPS_PATH="$(DEPS_PATH)" -C $(PS_PATH) ps

$(DMLC_CORE)/libdmlc.a: DMLCCORE

DMLCCORE:
	+ cd $(DMLC_CORE); $(MAKE) libdmlc.a USE_SSE=$(USE_SSE) config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

lib/libtvm_runtime.so:
	echo "Compile TVM"
	@mkdir -p $(@D)
	[ -e $(LLVM_PATH)/bin/llvm-config ] || sh $(ROOTDIR)/contrib/tvmop/prepare_tvm.sh; \
	cd $(TVM_PATH)/build; \
	cmake -DUSE_LLVM="$(LLVM_PATH)/bin/llvm-config" \
		  -DUSE_SORT=OFF -DUSE_CUDA=$(TVM_USE_CUDA) -DUSE_CUDNN=OFF -DUSE_OPENMP=ON ..; \
	$(MAKE) VERBOSE=1; \
	mkdir -p $(ROOTDIR)/lib; \
	cp $(TVM_PATH)/build/libtvm_runtime.so $(ROOTDIR)/lib/libtvm_runtime.so; \
	ls $(ROOTDIR)/lib; \
	cd $(ROOTDIR)

TVM_OP_COMPILE_OPTIONS = -o $(ROOTDIR)/lib --config $(ROOTDIR)/lib/tvmop.conf
ifneq ($(CUDA_ARCH),)
	TVM_OP_COMPILE_OPTIONS += --cuda-arch "$(CUDA_ARCH)"
endif
lib/libtvmop.so: lib/libtvm_runtime.so $(wildcard contrib/tvmop/*/*.py contrib/tvmop/*.py)
	echo "Compile TVM operators"
	@mkdir -p $(@D)
	PYTHONPATH=$(TVM_PATH)/python:$(TVM_PATH)/topi/python:$(ROOTDIR)/contrib \
		LD_LIBRARY_PATH=$(ROOTDIR)/lib \
	    python3 $(ROOTDIR)/contrib/tvmop/compile.py $(TVM_OP_COMPILE_OPTIONS)

NNVM_INC = $(wildcard $(NNVM_PATH)/include/*/*.h)
NNVM_SRC = $(wildcard $(NNVM_PATH)/src/*/*/*.cc $(NNVM_PATH)/src/*/*.cc $(NNVM_PATH)/src/*.cc)
$(NNVM_PATH)/lib/libnnvm.a: $(NNVM_INC) $(NNVM_SRC)
	+ cd $(NNVM_PATH); $(MAKE) lib/libnnvm.a DMLC_CORE_PATH=$(DMLC_CORE); cd $(ROOTDIR)

bin/im2rec: tools/im2rec.cc $(ALLX_DEP)

$(BIN) :
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -std=c++11  -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

# CPP Package
ifeq ($(USE_CPP_PACKAGE), 1)
include cpp-package/cpp-package.mk
CFLAGS += -DMXNET_USE_CPP_PACKAGE=1
endif

include mkldnn.mk
include tests/cpp/unittest.mk

extra-packages: $(EXTRA_PACKAGES)

test: $(TEST)

lint: cpplint rcpplint jnilint pylint

cpplint:
	3rdparty/dmlc-core/scripts/lint.py mxnet cpp include src plugin cpp-package tests \
	--exclude_path src/operator/contrib/ctc_include include/mkldnn

pylint:
	python3 -m pylint --rcfile=$(ROOTDIR)/ci/other/pylintrc --ignore-patterns=".*\.so$$,.*\.dll$$,.*\.dylib$$" python/mxnet

# MXNet extension dynamically loading libraries
EXT_LIBS = build/libcustomop_lib.so build/libtransposecsr_lib.so build/libtransposerowsp_lib.so build/libsubgraph_lib.so build/libpass_lib.so
ifeq ($(USE_CUDA), 1)
        EXT_LIBS += build/libcustomop_gpu_lib.so
endif
extension_libs: $(EXT_LIBS)

build/libcustomop_lib.so:
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -std=c++11 example/extensions/lib_custom_op/gemm_lib.cc src/lib_api.cc -o $@ -I include
build/libcustomop_gpu_lib.so:
	@mkdir -p $(@D)
	$(NVCC) -shared -std=c++11 -Xcompiler -fPIC example/extensions/lib_custom_op/relu_lib.cu src/lib_api.cc -o $@ -I include
build/libsubgraph_lib.so:
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -std=c++11 example/extensions/lib_subgraph/subgraph_lib.cc src/lib_api.cc -o $@ -I include
build/libtransposecsr_lib.so:
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -std=c++11 example/extensions/lib_custom_op/transposecsr_lib.cc src/lib_api.cc -o $@ -I include
build/libtransposerowsp_lib.so:
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -std=c++11 example/extensions/lib_custom_op/transposerowsp_lib.cc src/lib_api.cc -o $@ -I include
build/libcustomop_gpu_lib.so:
	@mkdir -p $(@D)
	$(NVCC) -shared -std=c++11 -Xcompiler -fPIC example/extensions/lib_custom_op/relu_lib.cu src/lib_api.cc -o $@ -I include
build/libsubgraph_lib.so:
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -std=c++11 example/extensions/lib_subgraph/subgraph_lib.cc src/lib_api.cc -o $@ -I include
build/libpass_lib.so:
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -std=c++11 example/extensions/lib_pass/pass_lib.cc src/lib_api.cc -o $@ -I include

# Cython build
cython:
	cd python; $(PYTHON) setup.py build_ext --inplace --with-cython

cython3:
	cd python; python3 setup.py build_ext --inplace --with-cython

cyclean:
	rm -rf python/mxnet/*/*.so python/mxnet/*/*.cpp

scalaclean:
	(cd $(ROOTDIR)/scala-package && mvn clean)

scalapkg:
	(cd $(ROOTDIR)/scala-package && mvn install -DskipTests)

scalainstall:
	(cd $(ROOTDIR)/scala-package && mvn install)

scalaunittest:
	(cd $(ROOTDIR)/scala-package && mvn install)

scalaintegrationtest:
	(cd $(ROOTDIR)/scala-package && mvn integration-test -DskipTests=false)

jnilint:
	3rdparty/dmlc-core/scripts/lint.py mxnet-jnicpp cpp scala-package/native/src --exclude_path scala-package/native/src/main/native/org_apache_mxnet_native_c_api.h

rclean:
	$(RM) -r R-package/src/image_recordio.h R-package/NAMESPACE R-package/man R-package/R/mxnet_generated.R \
		R-package/inst R-package/src/*.o R-package/src/*.so mxnet_*.tar.gz

build/rat/apache-rat-0.13/apache-rat-0.13.jar:
	mkdir -p build/rat
	cd build/rat; \
	wget http://mirror.metrocast.net/apache//creadur/apache-rat-0.13/apache-rat-0.13-bin.zip; \
	unzip apache-rat-0.13-bin.zip;

ratcheck: build/rat/apache-rat-0.13/apache-rat-0.13.jar
	exec 5>&1; \
	RAT_JAR=build/rat/apache-rat-0.13/apache-rat-0.13.jar; \
	OUTPUT=$(java -jar $(RAT_JAR) -E tests/nightly/apache_rat_license_check/rat-excludes -d .|tee >(cat - >&5)); \
    ERROR_MESSAGE="Printing headers for text files without a valid license header"; \
    echo "-------Process The Output-------"; \
    if [[ $OUTPUT =~ $ERROR_MESSAGE ]]; then \
        echo "ERROR: RAT Check detected files with unknown licenses. Please fix and run test again!"; \
        exit 1; \
    else \
        echo "SUCCESS: There are no files with an Unknown License."; \
    fi


ifneq ($(EXTRA_OPERATORS),)
clean: rclean cyclean $(EXTRA_PACKAGES_CLEAN)
	$(RM) -r build lib bin deps *~ */*~ */*/*~ */*/*/*~
	(cd scala-package && mvn clean) || true
	cd $(DMLC_CORE); $(MAKE) clean; cd -
	cd $(PS_PATH); $(MAKE) clean; cd -
	cd $(NNVM_PATH); $(MAKE) clean; cd -
	cd $(TVM_PATH); $(MAKE) clean; cd -
	cd $(AMALGAMATION_PATH); $(MAKE) clean; cd -
	$(RM) -r  $(patsubst %, %/*.d, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.d, $(EXTRA_OPERATORS))
	$(RM) -r  $(patsubst %, %/*.o, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.o, $(EXTRA_OPERATORS))
else
clean: rclean mkldnn_clean cyclean testclean $(EXTRA_PACKAGES_CLEAN)
	$(RM) -r build lib bin *~ */*~ */*/*~ */*/*/*~
	(cd scala-package && mvn clean) || true
	cd $(DMLC_CORE); $(MAKE) clean; cd -
	cd $(PS_PATH); $(MAKE) clean; cd -
	cd $(NNVM_PATH); $(MAKE) clean; cd -
	cd $(TVM_PATH); $(MAKE) clean; cd -
	cd $(AMALGAMATION_PATH); $(MAKE) clean; cd -
endif

clean_all: clean

-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
-include build/*/*/*/*.d
ifneq ($(EXTRA_OPERATORS),)
	-include $(patsubst %, %/*.d, $(EXTRA_OPERATORS)) $(patsubst %, %/*/*.d, $(EXTRA_OPERATORS))
endif
