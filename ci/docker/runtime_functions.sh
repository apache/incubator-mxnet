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
#
# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex

CI_CUDA_COMPUTE_CAPABILITIES="-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_70,code=sm_70"
CI_CMAKE_CUDA_ARCH="5.2 7.0"

clean_repo() {
    set -ex
    git clean -xfd
    git submodule foreach --recursive git clean -xfd
    git reset --hard
    git submodule foreach --recursive git reset --hard
    git submodule update --init --recursive
}

scala_prepare() {
    # Clean up maven logs
    export MAVEN_OPTS="-Dorg.slf4j.simpleLogger.log.org.apache.maven.cli.transfer.Slf4jMavenTransferListener=warn"
}

check_cython() {
    set -ex
    local is_cython_used=$(python3 <<EOF
import sys
import mxnet as mx
cython_ndarraybase = 'mxnet._cy3.ndarray'
print(mx.nd._internal.NDArrayBase.__module__ == cython_ndarraybase)
EOF
)

    if [ "${is_cython_used}" != "True" ]; then
        echo "ERROR: cython is not used."
        return 1
    else
        echo "NOTE: cython is used."
        return 0
    fi
}

build_wheel() {

    set -ex
    pushd .

    PYTHON_DIR=${1:-/work/mxnet/python}
    BUILD_DIR=${2:-/work/build}

    # build

    export MXNET_LIBRARY_PATH=${BUILD_DIR}/libmxnet.so

    cd ${PYTHON_DIR}
    python3 setup.py bdist_wheel

    # repackage

    # Fix pathing issues in the wheel.  We need to move libmxnet.so from the data folder to the
    # mxnet folder, then repackage the wheel.
    WHEEL=`readlink -f dist/*.whl`
    TMPDIR=`mktemp -d`
    unzip -d ${TMPDIR} ${WHEEL}
    rm ${WHEEL}
    cd ${TMPDIR}
    mv *.data/data/mxnet/libmxnet.so mxnet
    zip -r ${WHEEL} .
    cp ${WHEEL} ${BUILD_DIR}
    rm -rf ${TMPDIR}

    popd
}

gather_licenses() {
    mkdir -p licenses

    cp tools/dependencies/LICENSE.binary.dependencies licenses/
    cp NOTICE licenses/
    cp LICENSE licenses/
    cp DISCLAIMER licenses/
}

# Compiles the dynamic mxnet library
# Parameters:
# $1 -> mxnet_variant: the mxnet variant to build, e.g. cpu, native, cu101, cu102, etc.
build_dynamic_libmxnet() {
    set -ex

    local mxnet_variant=${1:?"This function requires a mxnet variant as the first argument"}

    # relevant licenses will be placed in the licenses directory
    gather_licenses

    cd /work/build
    source /opt/rh/devtoolset-8/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    if [[ ${mxnet_variant} = "cpu" ]]; then
        cmake -DUSE_BLAS=Open \
            -DUSE_ONEDNN=ON \
            -DUSE_CUDA=OFF \
            -G Ninja /work/mxnet
    elif [[ ${mxnet_variant} = "native" ]]; then
        cmake -DUSE_BLAS=Open \
            -DUSE_ONEDNN=OFF \
            -DUSE_CUDA=OFF \
            -G Ninja /work/mxnet
    elif [[ ${mxnet_variant} =~ cu[0-9]+$ ]]; then
        cmake -DUSE_BLAS=Open \
            -DUSE_ONEDNN=ON \
            -DUSE_DIST_KVSTORE=ON \
            -DUSE_CUDA=ON \
            -G Ninja /work/mxnet
    else
        echo "Error: Unrecognized mxnet variant '${mxnet_variant}'"
        exit 1
    fi
    ninja
}

build_jetson() {
    set -ex
    cd /work/build
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=ON \
        -DMXNET_CUDA_ARCH="5.2" \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_LAPACK=OFF \
        -DUSE_BLAS=Open \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja /work/mxnet
    ninja
    build_wheel
}

#
# ARM builds
#

build_armv6() {
    set -ex
    cd /work/build

    # We do not need OpenMP, since most armv6 systems have only 1 core

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_LAPACK=OFF \
        -DUSE_BLAS=Open \
        -DBUILD_CPP_EXAMPLES=OFF \
        -G Ninja /work/mxnet

    ninja
    build_wheel
}

build_armv7() {
    set -ex
    cd /work/build

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_LAPACK=OFF \
        -DUSE_BLAS=Open \
        -DBUILD_CPP_EXAMPLES=OFF \
        -G Ninja /work/mxnet

    ninja
    build_wheel
}

build_armv8() {
    cd /work/build
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_LAPACK=OFF \
        -DUSE_BLAS=Open \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja /work/mxnet
    ninja
    build_wheel
}


#
# ANDROID builds
#

build_android_armv7() {
    set -ex
    cd /work/build
    # ANDROID_ABI and ANDROID_STL are options of the CMAKE_TOOLCHAIN_FILE
    # provided by Android NDK
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DANDROID_ABI="armeabi-v7a" \
        -DANDROID_STL="c++_shared" \
        -DUSE_CUDA=OFF \
        -DUSE_LAPACK=OFF \
        -DUSE_BLAS=Open \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_android_armv8() {
    set -ex
    cd /work/build
    # ANDROID_ABI and ANDROID_STL are options of the CMAKE_TOOLCHAIN_FILE
    # provided by Android NDK
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_STL="c++_shared" \
        -DUSE_CUDA=OFF \
        -DUSE_LAPACK=OFF \
        -DUSE_BLAS=Open \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
        -G Ninja /work/mxnet
    ninja
}

build_centos7_cpu() {
    set -ex
    cd /work/build
    source /opt/rh/devtoolset-7/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_ONEDNN=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DUSE_CUDA=OFF \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -DUSE_INT64_TENSOR_SIZE=OFF \
        -DUSE_BLAS=Open \
        -G Ninja /work/mxnet
    ninja
}

build_centos7_onednn() {
    set -ex
    cd /work/build
    source /opt/rh/devtoolset-7/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    cmake -DUSE_BLAS=Open \
        -DUSE_ONEDNN=ON \
        -DUSE_CUDA=OFF \
        -DUSE_INT64_TENSOR_SIZE=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_centos7_gpu() {
    set -ex
    cd /work/build
    source /opt/rh/devtoolset-7/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_BLAS=Open \
        -DUSE_ONEDNN=ON \
        -DUSE_CUDA=ON \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -DUSE_INT64_TENSOR_SIZE=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu() {
    build_ubuntu_cpu_openblas
}

build_ubuntu_cpu_openblas() {
    set -ex
    cd /work/build
    CXXFLAGS="-Wno-error=strict-overflow" CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_TVM_OP=ON \
        -DUSE_BLAS=Open \
        -DUSE_ONEDNN=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_CYTHON_MODULES=ON \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_mkl() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=OFF \
        -DUSE_ONEDNN=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_MKL_LAYERNORM=ON \
        -DUSE_BLAS=MKL \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -GNinja /work/mxnet
    ninja
}

build_ubuntu_cpu_cmake_debug() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE=Debug \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_BLAS=Open \
        -DUSE_OPENMP=OFF \
        -DUSE_OPENCV=ON \
        -DUSE_SIGNAL_HANDLER=ON \
        -G Ninja \
        /work/mxnet
    ninja
}

build_ubuntu_cpu_cmake_no_tvm_op() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=OFF \
        -DUSE_BLAS=Open \
        -DUSE_OPENMP=OFF \
        -DUSE_OPENCV=ON \
        -DUSE_SIGNAL_HANDLER=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja \
        /work/mxnet

    ninja
}

build_ubuntu_cpu_cmake_asan() {
    set -ex

    cd /work/build
    cmake \
        -DUSE_CUDA=OFF \
        -DUSE_BLAS=Open \
        -DUSE_ONEDNN=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_OPENCV=OFF \
        -DCMAKE_BUILD_TYPE=Debug \
        -DUSE_GPERFTOOLS=OFF \
        -DUSE_JEMALLOC=OFF \
        -DUSE_ASAN=ON \
        /work/mxnet
    make -j $(nproc) mxnet
}

build_ubuntu_cpu_gcc8_werror() {
    set -ex
    cd /work/build
    CC=gcc-8 CXX=g++-8 cmake \
        -DUSE_BLAS=Open \
        -DUSE_CUDA=OFF \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -GNinja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang10_werror() {
    set -ex
    cd /work/build
    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_BLAS=Open \
       -DUSE_CUDA=OFF \
       -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
       -GNinja /work/mxnet
    ninja
}

build_ubuntu_gpu_clang10_werror() {
    set -ex
    cd /work/build
    # Disable cpp package as OpWrapperGenerator.py dlopens libmxnet.so,
    # requiring presence of cuda driver libraries that are missing on CI host
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs
    # Workaround https://github.com/thrust/thrust/issues/1072
    # Can be deleted on Cuda 11
    export CXXFLAGS="-I/usr/local/thrust"

    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_BLAS=Open \
       -DUSE_CUDA=ON \
       -DUSE_NVML=OFF \
       -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
       -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
       -GNinja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang6() {
    set -ex
    cd /work/build
    export OpenBLAS_HOME=/usr/local/openblas-clang/
    CXX=clang++-6.0 CC=clang-6.0 cmake \
        -DUSE_BLAS=Open \
        -DUSE_ONEDNN=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang100() {
    set -ex
    cd /work/build
    export OpenBLAS_HOME=/usr/local/openblas-clang/
    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_BLAS=Open \
       -DUSE_ONEDNN=OFF \
       -DUSE_CUDA=OFF \
       -DUSE_OPENMP=ON \
       -DUSE_DIST_KVSTORE=ON \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang_tidy() {
    set -ex
    cd /work/build
    export OpenBLAS_HOME=/usr/local/openblas-clang/
    # TODO(leezu) USE_OPENMP=OFF 3rdparty/dmlc-core/CMakeLists.txt:79 broken?
    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_BLAS=Open \
       -DUSE_ONEDNN=OFF \
       -DUSE_CUDA=OFF \
       -DUSE_OPENMP=OFF \
       -DCMAKE_BUILD_TYPE=Debug \
       -DUSE_DIST_KVSTORE=ON \
       -DCMAKE_CXX_CLANG_TIDY=clang-tidy-10 \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang6_onednn() {
    set -ex
    cd /work/build
    export OpenBLAS_HOME=/usr/local/openblas-clang/
    CXX=clang++-6.0 CC=clang-6.0 cmake \
       -DUSE_BLAS=Open \
       -DUSE_ONEDNN=ON \
       -DUSE_CUDA=OFF \
       -DUSE_OPENMP=OFF \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang100_onednn() {
    set -ex
    cd /work/build
    export OpenBLAS_HOME=/usr/local/openblas-clang/
    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_BLAS=Open \
       -DUSE_ONEDNN=ON \
       -DUSE_CUDA=OFF \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_onednn() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_TVM_OP=ON \
        -DUSE_BLAS=Open \
        -DUSE_ONEDNN=ON \
        -DUSE_CUDA=OFF \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_onednn_mkl() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=OFF \
        -DUSE_ONEDNN=ON \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_BLAS=MKL \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -GNinja /work/mxnet
    ninja
}

build_ubuntu_gpu_tensorrt() {

    set -ex

    export CC=gcc-7
    export CXX=g++-7

    cd /work/build
    cmake -DUSE_CUDA=1                            \
          -DUSE_CUDNN=1                           \
          -DUSE_OPENCV=1                          \
          -DONNX_NAMESPACE=onnx                   \
          -DUSE_TENSORRT=1                        \
          -DUSE_OPENMP=0                          \
          -DUSE_BLAS=Open                         \
          -DUSE_ONEDNN=0                          \
          -DUSE_NVML=OFF                          \
          -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
          -G Ninja                                \
          /work/mxnet

    ninja

    mkdir -p /work/mxnet/lib/
    cp 3rdparty/onnx-tensorrt/third_party/onnx/*.so /work/mxnet/lib/
    cp -L 3rdparty/onnx-tensorrt/libnvonnxparser.so* /work/mxnet/lib/
}

build_ubuntu_gpu_onednn() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_BLAS=Open \
        -DUSE_CUDA=ON \
        -DUSE_NVML=OFF \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_gpu_onednn_nocudnn() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_BLAS=Open \
        -DUSE_CUDA=ON \
        -DUSE_NVML=OFF \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_CUDNN=OFF \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_gpu() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_CUDA=ON \
        -DUSE_NVML=OFF \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_CUDNN=ON \
        -DUSE_BLAS=Open \
        -DUSE_ONEDNN=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_CYTHON_MODULES=ON \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_gpu_debug() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE=Debug \
        -DUSE_CUDA=ON \
        -DUSE_NVML=OFF \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_CUDNN=ON \
        -DUSE_BLAS=Open \
        -DUSE_ONEDNN=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_CYTHON_MODULES=ON \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_large_tensor() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=OFF                          \
        -DUSE_CUDNN=OFF                         \
        -DUSE_BLAS=Open                         \
        -DUSE_ONEDNN=ON                         \
        -G Ninja                                \
        /work/mxnet

    ninja
}

build_ubuntu_gpu_large_tensor() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=ON                           \
        -DUSE_CUDNN=ON                          \
        -DUSE_NVML=OFF                          \
        -DUSE_BLAS=Open                         \
        -DUSE_ONEDNN=ON                         \
        -DUSE_DIST_KVSTORE=ON                   \
        -DCMAKE_BUILD_TYPE=Release              \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -G Ninja                                \
        /work/mxnet

    ninja
}

# Testing

sanity_check() {
    set -ex
    sanity_license
    sanity_python
    sanity_cpp
}

sanity_license() {
    set -ex
    tools/license_header.py check
}

sanity_cpp() {
    set -ex
    3rdparty/dmlc-core/scripts/lint.py mxnet cpp include src plugin tests --exclude_path src/operator/contrib/ctc_include include/onednn
}

sanity_python() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    python3 -m pylint --rcfile=ci/other/pylintrc --ignore-patterns=".*\.so$$,.*\.dll$$,.*\.dylib$$" python/mxnet
    OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -n 4 tests/tutorials/test_sanity_tutorials.py
}

# Tests libmxnet
# Parameters:
# $1 -> mxnet_variant: The variant of the libmxnet.so library
cd_unittest_ubuntu() {
    set -ex
    source /opt/rh/rh-python36/enable
    export PYTHONPATH=./python/
    export MXNET_ONEDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export CD_JOB=1 # signal this is a CD run so any unecessary tests can be skipped
    export DMLC_LOG_STACK_TRACE_DEPTH=100

    local mxnet_variant=${1:?"This function requires a mxnet variant as the first argument"}

    OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -n 4 --durations=50 --verbose tests/python/unittest
    pytest -m 'serial' --durations=50 --verbose tests/python/unittest

    # https://github.com/apache/incubator-mxnet/issues/11801
    # if [[ ${mxnet_variant} = "cpu" ]] || [[ ${mxnet_variant} = "mkl" ]]; then
        # integrationtest_ubuntu_cpu_dist_kvstore
    # fi

    if [[ ${mxnet_variant} = cu* ]]; then
        MXNET_GPU_MEM_POOL_TYPE=Unpooled \
        MXNET_ENGINE_TYPE=NaiveEngine \
            OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --verbose tests/python/gpu
        MXNET_GPU_MEM_POOL_TYPE=Unpooled \
            OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'not test_operator and not test_amp_init.py' -n 4 --durations=50 --verbose tests/python/gpu
        pytest -m 'serial' --durations=50 --verbose tests/python/gpu
        pytest --durations=50 --verbose tests/python/gpu/test_amp_init.py

        # TODO(szha): fix and reenable the hanging issue. tracked in #18098
        # integrationtest_ubuntu_gpu_dist_kvstore
        # TODO(eric-haibin-lin): fix and reenable
        # integrationtest_ubuntu_gpu_byteps
    fi

    if [[ ${mxnet_variant} = *mkl ]]; then
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -n 4 --durations=50 --verbose tests/python/mkl
    fi
}

unittest_ubuntu_python3_cpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_ONEDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'not test_operator' -n 4 --durations=50 --cov-report xml:tests_unittest.xml --verbose tests/python/unittest
    MXNET_ENGINE_TYPE=NaiveEngine \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --cov-report xml:tests_unittest.xml --cov-append --verbose tests/python/unittest
    pytest -m 'serial' --durations=50 --cov-report xml:tests_unittest.xml --cov-append --verbose tests/python/unittest
}

unittest_ubuntu_python3_cpu_onednn() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_ONEDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'not test_operator' -n 4 --durations=50 --cov-report xml:tests_unittest.xml --verbose tests/python/unittest
    MXNET_ENGINE_TYPE=NaiveEngine \
                     OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --cov-report xml:tests_unittest.xml --cov-append --verbose tests/python/unittest
    pytest -m 'serial' --durations=50 --cov-report xml:tests_unittest.xml --cov-append --verbose tests/python/unittest
    pytest --durations=50 --cov-report xml:tests_mkl.xml --verbose tests/python/mkl
}

unittest_ubuntu_python3_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_ONEDNN_DEBUG=0 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    MXNET_GPU_MEM_POOL_TYPE=Unpooled \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'not test_operator and not test_amp_init.py' -n 4 --durations=50 --cov-report xml:tests_gpu.xml --verbose tests/python/gpu
    MXNET_GPU_MEM_POOL_TYPE=Unpooled \
    MXNET_ENGINE_TYPE=NaiveEngine \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    pytest -m 'serial' --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    pytest --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu/test_amp_init.py
}

unittest_ubuntu_python3_gpu_cython() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_ONEDNN_DEBUG=1 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export MXNET_ENABLE_CYTHON=1
    export MXNET_ENFORCE_CYTHON=1
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    check_cython
    MXNET_GPU_MEM_POOL_TYPE=Unpooled \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'not test_operator and not test_amp_init.py' -n 4 --durations=50 --cov-report xml:tests_gpu.xml --verbose tests/python/gpu
    MXNET_GPU_MEM_POOL_TYPE=Unpooled \
    MXNET_ENGINE_TYPE=NaiveEngine \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    pytest -m 'serial' --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    pytest --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu/test_amp_init.py
}

unittest_ubuntu_python3_gpu_nocudnn() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_OFF_TEST_ONLY=true
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    MXNET_GPU_MEM_POOL_TYPE=Unpooled \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'not test_operator and not test_amp_init.py' -n 4 --durations=50 --cov-report xml:tests_gpu.xml --verbose tests/python/gpu
    MXNET_GPU_MEM_POOL_TYPE=Unpooled \
    MXNET_ENGINE_TYPE=NaiveEngine \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    pytest -m 'serial' --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    pytest --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu/test_amp_init.py
}

unittest_cpp() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    build/tests/mxnet_unit_tests
}

unittest_centos7_cpu() {
    set -ex
    source /opt/rh/rh-python36/enable
    cd /work/mxnet
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    OMP_NUM_THREADS=$(expr $(nproc) / 4) python -m pytest -m 'not serial' -k 'not test_operator' -n 4 --durations=50 --cov-report xml:tests_unittest.xml --verbose tests/python/unittest
    MXNET_ENGINE_TYPE=NaiveEngine \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) python -m pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --cov-report xml:tests_unittest.xml --cov-append --verbose tests/python/unittest
    python -m pytest -m 'serial' --durations=50 --cov-report xml:tests_unittest.xml --cov-append --verbose tests/python/unittest
    OMP_NUM_THREADS=$(expr $(nproc) / 4) python -m pytest -n 4 --durations=50 --cov-report xml:tests_train.xml --verbose tests/python/train
}

unittest_centos7_gpu() {
    set -ex
    source /opt/rh/rh-python36/enable
    cd /work/mxnet
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    MXNET_GPU_MEM_POOL_TYPE=Unpooled \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'not test_operator and not test_amp_init.py' -n 4 --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    MXNET_GPU_MEM_POOL_TYPE=Unpooled \
    MXNET_ENGINE_TYPE=NaiveEngine \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    pytest -m 'serial' --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu
    pytest --durations=50 --cov-report xml:tests_gpu.xml --cov-append --verbose tests/python/gpu/test_amp_init.py
}

integrationtest_ubuntu_cpu_onnx() {
	set -ex
	export PYTHONPATH=./python/
	export MXNET_SUBGRAPH_VERBOSE=0
	export DMLC_LOG_STACK_TRACE_DEPTH=100
	python3 tests/python/unittest/onnx/backend_test.py
	#OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -n 4 tests/python/unittest/onnx/mxnet_export_test.py
	#OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -n 4 tests/python/unittest/onnx/test_models.py
	#OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -n 4 tests/python/unittest/onnx/test_node.py
	#OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -n 4 tests/python/unittest/onnx/test_onnxruntime.py
}

integrationtest_ubuntu_cpu_dist_kvstore() {
    set -ex
    pushd .
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_USE_OPERATOR_TUNING=0
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    cd tests/nightly/
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=gluon_step_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=gluon_sparse_step_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=invalid_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=gluon_type_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --no-multiprecision
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=compressed_cpu_1bit
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=compressed_cpu_1bit --no-multiprecision
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=compressed_cpu_2bit
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=compressed_cpu_2bit --no-multiprecision
    python3 ../../tools/launch.py -n 3 --launcher local python3 test_server_profiling.py
    popd
}

integrationtest_ubuntu_gpu_dist_kvstore() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    pushd .
    cd /work/mxnet/python
    pip3 install -e .
    pip3 install --no-cache-dir horovod
    cd /work/mxnet/tests/nightly
    ./test_distributed_training-gpu.sh
    popd
}

integrationtest_ubuntu_gpu_byteps() {
    set -ex
    pushd .
    export PYTHONPATH=$PWD/python/
    export BYTEPS_WITHOUT_PYTORCH=1
    export BYTEPS_WITHOUT_TENSORFLOW=1
    pip3 install byteps==0.2.3 --user
    git clone -b v0.2.3 https://github.com/bytedance/byteps ~/byteps
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    cd tests/nightly/

    export NVIDIA_VISIBLE_DEVICES=0
    export DMLC_WORKER_ID=0 # your worker id
    export DMLC_NUM_WORKER=1 # one worker
    export DMLC_ROLE=worker

    # the following value does not matter for non-distributed jobs
    export DMLC_NUM_SERVER=1
    export DMLC_PS_ROOT_URI=0.0.0.127
    export DMLC_PS_ROOT_PORT=1234

    python3 ~/byteps/launcher/launch.py python3 dist_device_sync_kvstore_byteps.py

    popd
}


test_ubuntu_cpu_python3() {
    set -ex
    pushd .
    export MXNET_LIBRARY_PATH=/work/build/libmxnet.so
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    VENV=mxnet_py3_venv
    virtualenv -p `which python3` $VENV
    source $VENV/bin/activate

    cd /work/mxnet/python
    pip3 install -e .
    cd /work/mxnet
    OMP_NUM_THREADS=$(expr $(nproc) / 4) python3 -m pytest -m 'not serial' -k 'not test_operator' -n 4 --durations=50 --verbose tests/python/unittest
    MXNET_ENGINE_TYPE=NaiveEngine \
        OMP_NUM_THREADS=$(expr $(nproc) / 4) python3 -m pytest -m 'not serial' -k 'test_operator' -n 4 --durations=50 --verbose tests/python/unittest
    python3 -m pytest -m 'serial' --durations=50 --verbose tests/python/unittest

    popd
}

# QEMU based ARM tests
unittest_ubuntu_python3_arm() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_ONEDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    python3 -m pytest -n 2 --verbose tests/python/unittest/test_engine.py
}

# Functions that run the nightly Tests:

#Runs Apache RAT Check on MXNet Source for License Headers
test_rat_check() {
    set -e
    set -o pipefail
    pushd .

    cd /usr/local/src/apache-rat-0.13

    # Use shell number 5 to duplicate the log output. It get sprinted and stored in $OUTPUT at the same time https://stackoverflow.com/a/12451419
    exec 5>&1
    OUTPUT=$(java -jar apache-rat-0.13.jar -E /work/mxnet/rat-excludes -d /work/mxnet|tee >(cat - >&5))
    ERROR_MESSAGE="Printing headers for text files without a valid license header"


    echo "-------Process The Output-------"

    if [[ $OUTPUT =~ $ERROR_MESSAGE ]]; then
        echo "ERROR: RAT Check detected files with unknown licenses. Please fix and run test again!";
        exit 1
    else
        echo "SUCCESS: There are no files with an Unknown License.";
    fi
    popd
}

#Single Node KVStore Test
nightly_test_KVStore_singleNode() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    tests/nightly/test_kvstore.py
}

#Test Large Tensor Size
nightly_test_large_tensor() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    pytest -s --exitfirst --verbose --timeout=7200 tests/nightly/test_np_large_array.py
}

#Tests Model backwards compatibility on MXNet
nightly_model_backwards_compat_test() {
    set -ex
    export PYTHONPATH=/work/mxnet/python/
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    ./tests/nightly/model_backwards_compatibility_check/model_backward_compat_checker.sh
}

#Backfills S3 bucket with models trained on earlier versions of mxnet
nightly_model_backwards_compat_train() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    ./tests/nightly/model_backwards_compatibility_check/train_mxnet_legacy_models.sh
}

nightly_tutorial_test_ubuntu_python3_gpu() {
    set -ex
    cd /work/mxnet/docs
    export BUILD_VER=tutorial
    export MXNET_DOCS_BUILD_MXNET=0
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    make html
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TUTORIAL_TEST_KERNEL=python3
    cd /work/mxnet/tests/tutorials
    pytest --durations=50 --cov-report xml:tests_tutorials.xml --capture=no test_tutorials.py
}

nightly_estimator() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=100
    cd /work/mxnet/tests/nightly/estimator
    export PYTHONPATH=/work/mxnet/python/
    pytest test_estimator_cnn.py
    pytest test_sentiment_rnn.py
}

# For testing PRs
deploy_docs() {
    set -ex
    pushd .

    export CC="ccache gcc"
    export CXX="ccache g++"

    build_python_docs

    popd
}


build_docs_setup() {
    build_folder="docs/_build"
    mxnetlib_folder="/work/mxnet/lib"

    mkdir -p $build_folder
    mkdir -p $mxnetlib_folder
}

build_jekyll_docs() {
    set -ex

    pushd .
    build_docs_setup
    pushd docs/static_site
    make clean
    make html
    popd

    GZIP=-9 tar zcvf jekyll-artifacts.tgz -C docs/static_site/build html
    mv jekyll-artifacts.tgz docs/_build/
    popd
}


build_python_docs() {
    set -ex
    pushd .

    build_docs_setup

    pushd docs/python_docs
    python3 -m pip install -r requirements
    python3 -m pip install themes/mx-theme
    python3 -m pip install -e /work/mxnet/python --user

    export PATH=/home/jenkins_slave/.local/bin:$PATH

    pushd python
    make clean
    make html EVAL=0

    GZIP=-9 tar zcvf python-artifacts.tgz -C build/_build/html .
    popd

    mv python/python-artifacts.tgz /work/mxnet/docs/_build/
    popd

    popd
}


build_c_docs() {
    set -ex
    pushd .

    build_docs_setup
    doc_path="docs/cpp_docs"
    pushd $doc_path

    make clean
    make html

    doc_artifact="c-artifacts.tgz"
    GZIP=-9 tar zcvf $doc_artifact -C build/html/html .
    popd

    mv $doc_path/$doc_artifact docs/_build/

    popd
}


build_docs() {
    pushd docs/_build
    tar -xzf jekyll-artifacts.tgz
    python_doc_folder='html/api/python/docs'

    # Python has it's own landing page/site so we don't put it in /docs/api
    mkdir -p $python_doc_folder && tar -xzf python-artifacts.tgz --directory $python_doc_folder

     # check if .htaccess file exists
    if [ ! -f "html/.htaccess" ]; then
        echo "html/.htaccess file does not exist. Exiting 1"
        exit 1
    fi
    # get the version
    version=$(grep "RewriteRule" html/.htaccess | grep -E "versions\/[0-9]" | sed -nre 's/^[^0-9]*(([0-9]+\.)*[0-9]+).*/\1/p')
    # count how many versions are found
    lines=$(echo "$version" | wc -l)
    # check if multiple versions are found
    if [ "$lines" != "1" ]; then
        echo "multiple versions detected: $lines. Exiting 1"
        exit 1
    fi
    # check if no version is found
    if [ "$version" == "" ]; then
        echo "no version found. Exiting 1"
        exit 1
    fi
    # print the one and only default mxnet version
    echo "detected version is $version"
    # check if the artifacts for this version exist
    if [ -d "html/versions/$version/api" ]; then
        echo "html/versions/$version/api directory exists"
    else
        echo "html/versions/$version/api directory does not exist! Exiting 1"
        exit 1
    fi

    # copy the full site for this version to versions folder
    mkdir -p html/versions/master
    for f in 404.html api assets blog community ecosystem features feed.xml get_started index.html; do
        cp -r html/$f html/versions/master/
    done

    # clean up temp files
    find html -type f -name '.DS_Store' -delete

    # archive artifact
    GZIP=-9 tar -zcvf full_website.tgz -C html .
    popd
}

build_docs_beta() {
    pushd docs/_build
    tar -xzf jekyll-artifacts.tgz
    python_doc_folder="html/versions/$BRANCH/api/python/docs"
    mkdir -p $python_doc_folder && tar -xzf python-artifacts.tgz --directory $python_doc_folder
    GZIP=-9 tar -zcvf beta_website.tgz -C html .
    popd
}

push_docs() {
    folder_name=$1
    set -ex
    pip3 install --user awscli
    export PATH=~/.local/bin:$PATH
    pushd docs/_build
    tar -xzf full_website.tgz --strip-components 1
    # check if folder_name already exists in versions
    pushd versions
    if [ -d "$folder_name" ]; then
        echo "Folder $folder_name already exists in versions. Please double check the FOLDER_NAME variable in Jenkens pipeline"
        exit 1
    fi
    mv master $folder_name
    popd
    zip -r9 versions.zip versions/.
    # Upload versions folder
    aws s3 cp versions.zip s3://mxnet-website-static-artifacts --acl public-read
    # Backup versions folder with the latest version name
    backup_file="versions_backup_upto_$folder_name.zip"
    aws s3 cp s3://mxnet-website-static-artifacts/versions.zip s3://mxnet-website-static-artifacts/$backup_file --acl public-read
    popd
}

create_repo() {
   repo_folder=$1
   mxnet_url=$2
   git clone $mxnet_url $repo_folder --recursive
   echo "Adding MXNet upstream repo..."
   cd $repo_folder
   git remote add upstream https://github.com/apache/incubator-mxnet
   cd ..
}


refresh_branches() {
   repo_folder=$1
   cd $repo_folder
   git fetch
   git fetch upstream
   cd ..
}

checkout() {
   repo_folder=$1
   cd $repo_folder
   # Overriding configs later will cause a conflict here, so stashing...
   git stash
   # Fails to checkout if not available locally, so try upstream
   git checkout "$repo_folder" || git branch $repo_folder "upstream/$repo_folder" && git checkout "$repo_folder" || exit 1
   if [ $tag == 'master' ]; then
      git pull
      # master gets warnings as errors for Sphinx builds
      OPTS="-W"
      else
      OPTS=
   fi
   git submodule update --init --recursive
   cd ..
}

build_static_libmxnet() {
    set -ex
    pushd .
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    source tools/staticbuild/build.sh ${mxnet_variant}
    popd
}

# Tests CD PyPI packaging in CI
ci_package_pypi() {
    set -ex
    # copies onednn header files to 3rdparty/onednn/include/oneapi/dnnl/ as in CD
    mkdir -p 3rdparty/onednn/include/oneapi/dnnl
    cp include/onednn/oneapi/dnnl/dnnl_version.h 3rdparty/onednn/include/oneapi/dnnl/.
    cp include/onednn/oneapi/dnnl/dnnl_config.h 3rdparty/onednn/include/oneapi/dnnl/.
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    cd_package_pypi ${mxnet_variant}
    cd_integration_test_pypi
}

# Packages libmxnet into wheel file
cd_package_pypi() {
    set -ex
    pushd .
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    ./cd/python/pypi/pypi_package.sh ${mxnet_variant}
    popd
}

# Sanity checks wheel file
cd_integration_test_pypi() {
    set -ex
    source /opt/rh/rh-python36/enable

    # install mxnet wheel package
    pip3 install --user ./wheel_build/dist/*.whl

    # execute tests
    # TODO: Add tests (18549)
}

# Publishes wheel to PyPI
cd_pypi_publish() {
    set -ex
    pip3 install --user twine
    python3 ./cd/python/pypi/pypi_publish.py `readlink -f wheel_build/dist/*.whl`
}

cd_s3_publish() {
    set -ex
    pip3 install --upgrade --user awscli
    filepath=$(readlink -f wheel_build/dist/*.whl)
    filename=$(basename $filepath)
    variant=$(echo $filename | cut -d'-' -f1 | cut -d'_' -f2 -s)
    if [ -z "${variant}" ]; then
        variant="cpu"
    fi
    export PATH=/usr/local/bin:$PATH
    aws s3 cp ${filepath} s3://apache-mxnet/dist/python/${variant}/${filename} --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers full=id=43f628fab72838a4f0b929d7f1993b14411f4b0294b011261bc6bd3e950a6822
}

build_static_python_cpu() {
    set -ex
    pushd .
    export mxnet_variant=cpu
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    ./ci/publish/python/build.sh
    popd
}

build_static_python_cu102() {
    set -ex
    pushd .
    export mxnet_variant=cu102
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    ./ci/publish/python/build.sh
    popd
}

# artifact repository unit tests
test_artifact_repository() {
    set -ex
    pushd .
    cd cd/utils/
    OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -n 4 test_artifact_repository.py
    popd
}

##############################################################
# MAIN
#
# Run function passed as argument
set +x
if [ $# -gt 0 ]
then
    $@
else
    cat<<EOF

$0: Execute a function by passing it as an argument to the script:

Possible commands:

EOF
    declare -F | cut -d' ' -f3
    echo
fi
