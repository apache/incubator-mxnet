// -*- mode: groovy -*-

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// This file contains the steps that will be used in the
// Jenkins pipelines

utils = load('ci/Jenkinsfile_utils.groovy')

// mxnet libraries
mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, lib/libtvm_runtime.so, lib/libtvmop.so, lib/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, 3rdparty/dmlc-core/libdmlc.a, 3rdparty/tvm/nnvm/lib/libnnvm.a'
mx_lib_cython = 'lib/libmxnet.so, lib/libmxnet.a, lib/libtvm_runtime.so, lib/libtvmop.so, lib/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, 3rdparty/dmlc-core/libdmlc.a, 3rdparty/tvm/nnvm/lib/libnnvm.a, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so'

// mxnet cmake libraries, in cmake builds we do not produce a libnvvm static library by default.
mx_cmake_lib = 'build/libmxnet.so, build/libmxnet.a, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/3rdparty/dmlc-core/libdmlc.a, build/tests/mxnet_unit_tests, build/3rdparty/openmp/runtime/src/libomp.so'
mx_cmake_lib_no_tvm_op = 'build/libmxnet.so, build/libmxnet.a, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, build/3rdparty/dmlc-core/libdmlc.a, build/tests/mxnet_unit_tests, build/3rdparty/openmp/runtime/src/libomp.so'
mx_cmake_lib_cython = 'build/libmxnet.so, build/libmxnet.a, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/3rdparty/dmlc-core/libdmlc.a, build/tests/mxnet_unit_tests, build/3rdparty/openmp/runtime/src/libomp.so, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so'
// mxnet cmake libraries, in cmake builds we do not produce a libnvvm static library by default.
mx_cmake_lib_debug = 'build/libmxnet.so, build/libmxnet.a, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, build/3rdparty/dmlc-core/libdmlc.a, build/tests/mxnet_unit_tests'
mx_cmake_mkldnn_lib = 'build/libmxnet.so, build/libmxnet.a, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/3rdparty/dmlc-core/libdmlc.a, build/tests/mxnet_unit_tests, build/3rdparty/openmp/runtime/src/libomp.so'
mx_mkldnn_lib = 'lib/libmxnet.so, lib/libmxnet.a, lib/libtvm_runtime.so, lib/libtvmop.so, lib/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, 3rdparty/dmlc-core/libdmlc.a, 3rdparty/tvm/nnvm/lib/libnnvm.a'
mx_tensorrt_lib = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, lib/libnvonnxparser.so*, lib/libonnx_proto.so, lib/libonnx.so'
mx_lib_cpp_examples = 'lib/libmxnet.so, lib/libmxnet.a, lib/libtvm_runtime.so, lib/libtvmop.so, lib/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, 3rdparty/dmlc-core/libdmlc.a, 3rdparty/tvm/nnvm/lib/libnnvm.a, 3rdparty/ps-lite/build/libps.a, deps/lib/libprotobuf-lite.a, deps/lib/libzmq.a, build/cpp-package/example/*, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so'
mx_lib_cpp_capi = 'lib/libmxnet.so, lib/libmxnet.a, lib/libtvm_runtime.so, lib/libtvmop.so, lib/tvmop.conf, libsample_lib.so, lib/libmkldnn.so.1, lib/libmklml_intel.so, 3rdparty/dmlc-core/libdmlc.a, 3rdparty/tvm/nnvm/lib/libnnvm.a, 3rdparty/ps-lite/build/libps.a, deps/lib/libprotobuf-lite.a, deps/lib/libzmq.a, build/cpp-package/example/*, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so, build/tests/cpp/mxnet_unit_tests'
mx_lib_cpp_examples_no_tvm_op = 'lib/libmxnet.so, lib/libmxnet.a, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, 3rdparty/dmlc-core/libdmlc.a, 3rdparty/tvm/nnvm/lib/libnnvm.a, 3rdparty/ps-lite/build/libps.a, deps/lib/libprotobuf-lite.a, deps/lib/libzmq.a, build/cpp-package/example/*, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so'
mx_lib_cpp_examples_cpu = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/cpp-package/example/*'

// Python unittest for CPU
// Python 3
def python3_ut(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    utils.docker_run(docker_container_name, 'unittest_ubuntu_python3_cpu', false)
  }
}

// Python 3
def python3_ut_asan(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    utils.docker_run(docker_container_name, 'unittest_ubuntu_python3_cpu_asan', false)
  }
}

def python3_ut_mkldnn(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    utils.docker_run(docker_container_name, 'unittest_ubuntu_python3_cpu_mkldnn', false)
  }
}

// GPU test has two parts. 1) run unittest on GPU, 2) compare the results on
// both CPU and GPU
// Python 3
def python3_gpu_ut(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    utils.docker_run(docker_container_name, 'unittest_ubuntu_python3_gpu', true)
  }
}

// Python 3 NOCUDNN
def python3_gpu_ut_nocudnn(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    utils.docker_run(docker_container_name, 'unittest_ubuntu_python3_gpu_nocudnn', true)
  }
}

def python3_gpu_ut_cython(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    utils.docker_run(docker_container_name, 'unittest_ubuntu_python3_gpu_cython', true)
  }
}

//------------------------------------------------------------------------------------------

def compile_unix_cpu_openblas() {
    return ['CPU: Openblas': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-openblas') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_openblas', false)
            utils.pack_lib('cpu', mx_lib_cython)
          }
        }
      }
    }]
}

def compile_unix_openblas_debug_cpu() {
    return ['CPU: Openblas, cmake, debug': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-openblas') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_cmake_debug', false)
            utils.pack_lib('cpu_debug', mx_cmake_lib_debug)
          }
        }
      }
    }]
}

def compile_unix_openblas_cpu_no_tvm_op() {
    return ['CPU: Openblas, cmake, TVM_OP OFF': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-openblas-no-tvm-op') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_cmake_no_tvm_op', false)
            utils.pack_lib('cpu_openblas_no_tvm_op', mx_cmake_lib_no_tvm_op)
          }
        }
      }
    }]
}

def compile_unix_int64_cpu() {
    return ['CPU: USE_INT64_TENSOR_SIZE': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-int64') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_large_tensor', false)
            utils.pack_lib('ubuntu_cpu_int64', mx_cmake_lib)
          }
        }
      }
    }]
}

def compile_unix_int64_gpu() {
    return ['GPU: USE_INT64_TENSOR_SIZE': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/build-gpu-int64') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_cu101', 'build_ubuntu_gpu_large_tensor', false)
            utils.pack_lib('ubuntu_gpu_int64', mx_cmake_lib)
          }
        }
      }
    }]
}

def compile_unix_mkl_cpu() {
    return ['CPU: MKL': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-mkl') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_mkl', false)
            utils.pack_lib('cpu_mkl', mx_mkldnn_lib)
          }
        }
      }
    }]
}

def compile_unix_mkldnn_cpu() {
    return ['CPU: MKLDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-mkldnn-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_mkldnn', false)
            utils.pack_lib('mkldnn_cpu', mx_mkldnn_lib)
          }
        }
      }
    }]
}

def compile_unix_mkldnn_mkl_cpu() {
    return ['CPU: MKLDNN_MKL': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-mkldnn-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_mkldnn_mkl', false)
            utils.pack_lib('mkldnn_mkl_cpu', mx_mkldnn_lib)
          }
        }
      }
    }]
}

def compile_unix_mkldnn_gpu() {
    return ['GPU: MKLDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-mkldnn-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_build_cuda', 'build_ubuntu_gpu_mkldnn', false)
            utils.pack_lib('mkldnn_gpu', mx_mkldnn_lib)
          }
        }
      }
    }]
}

def compile_unix_mkldnn_nocudnn_gpu() {
    return ['GPU: MKLDNN_CUDNNOFF': {
       node(NODE_LINUX_CPU) {
         ws('workspace/build-mkldnn-gpu-nocudnn') {
           timeout(time: max_time, unit: 'MINUTES') {
             utils.init_git()
             utils.docker_run('ubuntu_build_cuda', 'build_ubuntu_gpu_mkldnn_nocudnn', false)
             utils.pack_lib('mkldnn_gpu_nocudnn', mx_mkldnn_lib)
           }
         }
       }
    }]
}

def compile_unix_full_gpu() {
    return ['GPU: CUDA10.1+cuDNN7': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_build_cuda', 'build_ubuntu_gpu_cuda101_cudnn7', false)
            utils.pack_lib('gpu', mx_lib_cpp_examples)
          }
        }
      }
    }]
}

def compile_unix_full_gpu_cu110() {
    return ['GPU: CUDA11.0+cuDNN8': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_build_cuda110', 'build_ubuntu_gpu_cuda110_cudnn8', false)
            utils.pack_lib('gpu_cu110', mx_lib_cpp_examples)
          }
        }
      }
    }]
}

def compile_unix_full_gpu_mkldnn_cpp_test() {
    return ['GPU: CUDA10.1+cuDNN7+MKLDNN+CPPTEST': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-gpu-mkldnn-cpp') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_build_cuda', 'build_ubuntu_gpu_cuda101_cudnn7_mkldnn_cpp_test', false)
            utils.pack_lib('gpu_mkldnn_cpp_test', mx_lib_cpp_capi)
          }
        }
      }
    }]
}

def compile_unix_cmake_mkldnn_gpu() {
    return ['GPU: CMake MKLDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cmake-mkldnn-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_cu101', 'build_ubuntu_gpu_cmake_mkldnn', false)
            utils.pack_lib('cmake_mkldnn_gpu', mx_cmake_mkldnn_lib)
          }
        }
      }
    }]
}

def compile_unix_cmake_gpu() {
    return ['GPU: CMake': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cmake-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_cu101', 'build_ubuntu_gpu_cmake', false)
            utils.pack_lib('cmake_gpu', mx_cmake_lib_cython)
          }
        }
      }
    }]
}

def compile_unix_cmake_gpu_no_rtc() {
    return ['GPU: CMake CUDA RTC OFF': {
        node(NODE_LINUX_CPU) {
            ws('workspace/build-cmake-gpu-no-rtc') {
                timeout(time: max_time, unit: 'MINUTES') {
                    utils.init_git()
                    utils.docker_run('ubuntu_gpu_cu101', 'build_ubuntu_gpu_cmake_no_rtc', false)
                }
            }
        }
    }]
}

def compile_unix_tensorrt_gpu() {
    return ['TensorRT': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-tensorrt') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_tensorrt', 'build_ubuntu_gpu_tensorrt', false)
            utils.pack_lib('tensorrt', mx_tensorrt_lib)
          }
        }
      }
    }]
}

def compile_centos7_cpu() {
    return ['CPU: CentOS 7': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-centos7-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_cpu', 'build_centos7_cpu', false)
            utils.pack_lib('centos7_cpu', mx_lib)
          }
        }
      }
    }]
}

def compile_centos7_cpu_mkldnn() {
    return ['CPU: CentOS 7 MKLDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-centos7-mkldnn') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_cpu', 'build_centos7_mkldnn', false)
            utils.pack_lib('centos7_mkldnn', mx_mkldnn_lib)
          }
        }
      }
    }]
}

def compile_centos7_gpu() {
    return ['GPU: CentOS 7': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-centos7-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_gpu', 'build_centos7_gpu', false)
            utils.pack_lib('centos7_gpu', mx_lib)
          }
        }
      }
    }]
}

def compile_unix_clang_3_9_cpu() {
    return ['CPU: Clang 3.9': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-clang39') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang39', false)
          }
        }
      }
    }]
}

def compile_unix_clang_6_cpu() {
    return ['CPU: Clang 6': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-clang60') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang60', false)
          }
        }
      }
    }]
}

def compile_unix_clang_tidy_cpu() {
    return ['CPU: Clang Tidy': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-clang60_tidy') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang_tidy', false)
          }
        }
      }
    }]
}

def compile_unix_clang_3_9_mkldnn_cpu() {
    return ['CPU: Clang 3.9 MKLDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-mkldnn-clang39') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang39_mkldnn', false)
            utils.pack_lib('mkldnn_cpu_clang3', mx_mkldnn_lib)
          }
        }
      }
    }]
}

def compile_unix_clang_6_mkldnn_cpu() {
    return ['CPU: Clang 6 MKLDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-mkldnn-clang60') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang60_mkldnn', false)
            utils.pack_lib('mkldnn_cpu_clang6', mx_mkldnn_lib)
          }
        }
      }
    }]
}

def compile_armv8_jetson_gpu() {
    return ['NVidia Jetson / ARMv8':{
      node(NODE_LINUX_CPU) {
        ws('workspace/build-jetson-armv8') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('jetson', 'build_jetson', false)
          }
        }
      }
    }]
}

def compile_armv6_cpu() {
    return ['ARMv6':{
      node(NODE_LINUX_CPU) {
        ws('workspace/build-ARMv6') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('armv6', 'build_armv6', false)
            utils.pack_lib('armv6', mx_cmake_lib)
          }
        }
      }
    }]
}

def compile_armv7_cpu() {
    return ['ARMv7':{
      node(NODE_LINUX_CPU) {
        ws('workspace/build-ARMv7') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('armv7', 'build_armv7', false)
            utils.pack_lib('armv7', mx_cmake_lib)
          }
        }
      }
    }]
}

def compile_armv8_cpu() {
    return ['ARMv8':{
      node(NODE_LINUX_CPU) {
        ws('workspace/build-ARMv8') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('armv8', 'build_armv8', false)
            utils.pack_lib('armv8', mx_cmake_lib)
          }
        }
      }
    }]
}

def compile_armv8_android_cpu() {
    return ['Android / ARMv8':{
      node(NODE_LINUX_CPU) {
        ws('workspace/android64') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('android_armv8', 'build_android_armv8', false)
          }
        }
      }
    }]
}

def compile_armv7_android_cpu() {
    return ['Android / ARMv7':{
      node(NODE_LINUX_CPU) {
        ws('workspace/androidv7') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('android_armv7', 'build_android_armv7', false)
          }
        }
      }
    }]
}

def compile_unix_asan_cpu() {
    return ['CPU: ASAN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-asan') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_cmake_asan', false)
            utils.pack_lib('cpu_asan', mx_lib_cpp_examples_cpu)
          }
        }
      }
    }]
}

def compile_unix_amalgamation_min() {
    return ['Amalgamation MIN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/amalgamationmin') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_amalgamation_min', false)
          }
        }
      }
    }]
}

def compile_unix_amalgamation() {
    return ['Amalgamation': {
      node(NODE_LINUX_CPU) {
        ws('workspace/amalgamation') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_amalgamation', false)
          }
        }
      }
    }]
}

def compile_windows_cpu() {
    return ['Build CPU windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_CPU'
            stash includes: 'windows_package.7z', name: 'windows_package_cpu'
          }
        }
      }
    }]
}

def compile_windows_cpu_mkldnn() {
    return ['Build CPU MKLDNN windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-cpu-mkldnn') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_CPU_MKLDNN'
            stash includes: 'windows_package.7z', name: 'windows_package_cpu_mkldnn'
          }
        }
      }
    }]
}

def compile_windows_cpu_mkldnn_mkl() {
    return ['Build CPU MKLDNN MKL windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-cpu-mkldnn-mkl') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_CPU_MKLDNN_MKL'
            stash includes: 'windows_package.7z', name: 'windows_package_cpu_mkldnn_mkl'
          }
        }
      }
    }]
}

def compile_windows_cpu_mkl() {
    return ['Build CPU MKL windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-cpu-mkl') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_CPU_MKL'
            stash includes: 'windows_package.7z', name: 'windows_package_cpu_mkl'
          }
        }
      }
    }]
}

def compile_windows_gpu() {
    return ['Build GPU windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
              utils.init_git_win()
              powershell 'py -3 ci/build_windows.py -f WIN_GPU'
              stash includes: 'windows_package.7z', name: 'windows_package_gpu'
          }
        }
      }
    }]
}

def compile_windows_gpu_mkldnn() {
    return ['Build GPU MKLDNN windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_GPU_MKLDNN'
            stash includes: 'windows_package.7z', name: 'windows_package_gpu_mkldnn'
          }
        }
      }
    }]
}

def test_static_scala_cpu() {
  return ['Static build CPU 14.04 Scala' : {
    node(NODE_LINUX_CPU) {
        ws('workspace/ut-publish-scala-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run("publish.ubuntu1404_cpu", 'build_static_scala_cpu', false)
          }
        }
    }
  }]
}

def test_static_python_cpu() {
  return ['Static build CPU 14.04 Python' : {
    node(NODE_LINUX_CPU) {
        ws('workspace/ut-publish-python-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run("publish.ubuntu1404_cpu", 'build_static_python_cpu', false)
          }
        }
    }
  }]
}

def test_static_python_cpu_cmake() {
    return ['Static build CPU 14.04 Python with CMake' : {
        node(NODE_LINUX_CPU) {
            ws('workspace/ut-publish-python-cpu') {
                timeout(time: max_time, unit: 'MINUTES') {
                    utils.init_git()
                    utils.docker_run("publish.ubuntu1404_cpu", 'build_static_python_cpu_cmake', false)
                }
            }
        }
    }]
}

def test_static_python_gpu() {
  return ['Static build GPU 14.04 Python' : {
    node(NODE_LINUX_GPU) {
        ws('workspace/ut-publish-python-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run("publish.ubuntu1404_gpu", 'build_static_python_cu101', true)
          }
        }
    }
  }]
}

def test_static_python_gpu_cmake() {
    return ['Static build GPU 14.04 Python' : {
        node(NODE_LINUX_GPU) {
            ws('workspace/ut-publish-python-gpu') {
                timeout(time: max_time, unit: 'MINUTES') {
                    utils.init_git()
                    utils.docker_run("publish.ubuntu1404_gpu", 'build_static_python_cu101_cmake', true)
                }
            }
        }
    }]
}

def test_unix_python3_cpu() {
    return ['Python3: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu') {
          try {
            utils.unpack_and_init('cpu', mx_lib)
            python3_ut('ubuntu_cpu')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_unittest.xml', 'nosetests_python3_cpu_unittest.xml')
            utils.collect_test_results_unix('nosetests_quantization.xml', 'nosetests_python3_cpu_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_memleak() {
    return ['Python3: Memory Leak': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu') {
          try {
            utils.unpack_and_init('cpu', mx_lib)
            python3_ut('ubuntu_cpu')
            utils.docker_run('ubuntu_valgrind', 'unittest_ubuntu_python3_quantization_gpu', true)
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_unittest.xml', 'nosetests_python3_cpu_unittest.xml')
            utils.collect_test_results_unix('nosetests_quantization.xml', 'nosetests_python3_cpu_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_mkl_cpu() {
    return ['Python3: MKL-CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu') {
          try {
            utils.unpack_and_init('cpu_mkl', mx_lib)
            python3_ut('ubuntu_cpu')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_unittest.xml', 'nosetests_python3_cpu_unittest.xml')
            utils.collect_test_results_unix('nosetests_quantization.xml', 'nosetests_python3_cpu_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_gpu() {
    return ['Python3: GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-gpu') {
          try {
            utils.unpack_and_init('gpu', mx_lib_cython)
            python3_gpu_ut_cython('ubuntu_gpu_cu101')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_gpu.xml', 'nosetests_python3_gpu.xml')
          }
        }
      }
    }]
}

def test_unix_python3_gpu_cu110() {
    return ['Python3+CUDA11.0: GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-gpu') {
          try {
            utils.unpack_and_init('gpu_cu110', mx_lib_cython)
            python3_gpu_ut_cython('ubuntu_gpu_cu110')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_gpu.xml', 'nosetests_python3_gpu.xml')
          }
        }
      }
    }]
}

def test_unix_python3_quantize_gpu() {
    return ['Python3: Quantize GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-quantize-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            try {
              utils.unpack_and_init('gpu', mx_lib)
              utils.docker_run('ubuntu_gpu_cu101', 'unittest_ubuntu_python3_quantization_gpu', true)
              utils.publish_test_coverage()
            } finally {
              utils.collect_test_results_unix('nosetests_quantization_gpu.xml', 'nosetests_python3_quantize_gpu.xml')
            }
          }
        }
      }
    }]
}

def test_unix_python3_quantize_gpu_cu110() {
    return ['Python3+CUDA11.0: Quantize GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-quantize-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            try {
              utils.unpack_and_init('gpu_cu110', mx_lib)
              utils.docker_run('ubuntu_gpu_cu110', 'unittest_ubuntu_python3_quantization_gpu', true)
              utils.publish_test_coverage()
            } finally {
              utils.collect_test_results_unix('nosetests_quantization_gpu.xml', 'nosetests_python3_quantize_gpu.xml')
            }
          }
        }
      }
    }]
}

def test_unix_python3_debug_cpu() {
    return ['Python3: CPU debug': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu-debug') {
          try {
            utils.unpack_and_init('cpu_debug', mx_cmake_lib_debug)
            python3_ut('ubuntu_cpu')
          } finally {
            utils.collect_test_results_unix('nosetests_unittest.xml', 'nosetests_python3_cpu_debug_unittest.xml')
            utils.collect_test_results_unix('nosetests_quantization.xml', 'nosetests_python3_cpu_debug_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_cpu_no_tvm_op() {
    return ['Python3: CPU TVM_OP OFF': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu-no-tvm-op') {
          try {
            utils.unpack_and_init('cpu_openblas_no_tvm_op', mx_cmake_lib_no_tvm_op)
            python3_ut('ubuntu_cpu')
          } finally {
            utils.collect_test_results_unix('nosetests_unittest.xml', 'nosetests_python3_cpu_no_tvm_op_unittest.xml')
            utils.collect_test_results_unix('nosetests_quantization.xml', 'nosetests_python3_cpu_no_tvm_op_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_mkldnn_cpu() {
    return ['Python3: MKLDNN-CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-mkldnn-cpu') {
          try {
            utils.unpack_and_init('mkldnn_cpu', mx_mkldnn_lib)
            python3_ut_mkldnn('ubuntu_cpu')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_unittest.xml', 'nosetests_python3_mkldnn_cpu_unittest.xml')
            utils.collect_test_results_unix('nosetests_mkl.xml', 'nosetests_python3_mkldnn_cpu_mkl.xml')
          }
        }
      }
    }]
}

def test_unix_python3_mkldnn_mkl_cpu() {
    return ['Python3: MKLDNN-MKL-CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-mkldnn-mkl-cpu') {
          try {
            utils.unpack_and_init('mkldnn_mkl_cpu', mx_mkldnn_lib)
            python3_ut_mkldnn('ubuntu_cpu')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_unittest.xml', 'nosetests_python3_mkldnn_cpu_unittest.xml')
            utils.collect_test_results_unix('nosetests_mkl.xml', 'nosetests_python3_mkldnn_cpu_mkl.xml')
          }
        }
      }
    }]
}

def test_unix_python3_mkldnn_gpu() {
    return ['Python3: MKLDNN-GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-mkldnn-gpu') {
          try {
            utils.unpack_and_init('mkldnn_gpu', mx_mkldnn_lib)
            python3_gpu_ut('ubuntu_gpu_cu101')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_gpu.xml', 'nosetests_python3_mkldnn_gpu.xml')
          }
        }
      }
    }]
}

def test_unix_python3_mkldnn_nocudnn_gpu() {
    return ['Python3: MKLDNN-GPU-NOCUDNN': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-mkldnn-gpu-nocudnn') {
          try {
            utils.unpack_and_init('mkldnn_gpu_nocudnn', mx_mkldnn_lib)
            python3_gpu_ut_nocudnn('ubuntu_gpu_cu101')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('nosetests_gpu.xml', 'nosetests_python3_mkldnn_gpu_nocudnn.xml')
          }
        }
      }
    }]
}

def test_unix_python3_tensorrt_gpu() {
    return ['Python3: TensorRT GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/build-tensorrt') {
          timeout(time: max_time, unit: 'MINUTES') {
            try {
              utils.unpack_and_init('tensorrt', mx_tensorrt_lib)
              utils.docker_run('ubuntu_gpu_tensorrt', 'unittest_ubuntu_tensorrt_gpu', true)
              utils.publish_test_coverage()
            } finally {
              utils.collect_test_results_unix('nosetests_tensorrt.xml', 'nosetests_python3_tensorrt_gpu.xml')
            }
          }
        }
      }
    }]
}

def test_unix_python3_integration_gpu() {
    return ['Python Integration GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/it-python-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('gpu', mx_lib)
            utils.docker_run('ubuntu_gpu_cu101', 'integrationtest_ubuntu_gpu_python', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_caffe_gpu() {
    return ['Caffe GPU': {
        node(NODE_LINUX_GPU) {
            ws('workspace/it-caffe') {
            timeout(time: max_time, unit: 'MINUTES') {
                utils.init_git()
                utils.unpack_lib('gpu', mx_lib)
                utils.docker_run('ubuntu_gpu_cu101', 'integrationtest_ubuntu_gpu_caffe', true)
                utils.publish_test_coverage()
            }
            }
        }
    }]
}

def test_unix_cpp_package_gpu() {
    return ['cpp-package GPU Makefile': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/it-cpp-package') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('gpu', mx_lib_cpp_examples)
            utils.docker_run('ubuntu_gpu_cu101', 'integrationtest_ubuntu_gpu_cpp_package', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_capi_cpp_package() {
    return ['capi-cpp-package GPU Makefile': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/it-capi-cpp-package') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('gpu_mkldnn_cpp_test', mx_lib_cpp_capi)
            utils.docker_run('ubuntu_gpu_cu101', 'integrationtest_ubuntu_gpu_capi_cpp_package', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_scala_cpu() {
    return ['Scala: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-scala-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'integrationtest_ubuntu_cpu_scala', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_scala_mkldnn_cpu(){
  return ['Scala: MKLDNN-CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-scala-mkldnn-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('mkldnn_cpu', mx_mkldnn_lib)
            utils.docker_run('ubuntu_cpu', 'integrationtest_ubuntu_cpu_scala', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_scala_gpu() {
    return ['Scala: GPU Makefile': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-scala-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('gpu', mx_lib)
            utils.docker_run('ubuntu_gpu_cu101', 'integrationtest_ubuntu_gpu_scala', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_clojure_cpu() {
    return ['Clojure: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-clojure-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'unittest_ubuntu_cpu_clojure', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_clojure_integration_cpu() {
    return ['Clojure: CPU Integration': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-clojure-integration-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'unittest_ubuntu_cpu_clojure_integration', false)
          }
        }
      }
    }]
}

def test_unix_r_cpu() {
    return ['R: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-r-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'unittest_ubuntu_cpu_R', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_r_mkldnn_cpu() {
    return ['R: MKLDNN-CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-r-mkldnn-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('mkldnn_cpu', mx_mkldnn_lib)
            utils.docker_run('ubuntu_cpu', 'unittest_ubuntu_minimal_R', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_perl_cpu() {
    return ['Perl: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-perl-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'unittest_ubuntu_cpugpu_perl', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_cpp_gpu() {
    return ['Cpp: GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-cpp-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cmake_gpu', mx_cmake_lib)
            utils.docker_run('ubuntu_gpu_cu101', 'unittest_cpp', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_cpp_mkldnn_gpu() {
    return ['Cpp: MKLDNN+GPU': {
      node(NODE_LINUX_GPU) {
        ws('workspace/ut-cpp-mkldnn-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cmake_mkldnn_gpu', mx_cmake_mkldnn_lib)
            utils.docker_run('ubuntu_gpu_cu101', 'unittest_cpp', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_cpp_cpu() {
    return ['Cpp: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-cpp-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu_debug', mx_cmake_lib_debug)
            utils.docker_run('ubuntu_cpu', 'unittest_cpp', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_perl_gpu() {
    return ['Perl: GPU Makefile': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-perl-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('gpu', mx_lib)
            utils.docker_run('ubuntu_gpu_cu101', 'unittest_ubuntu_cpugpu_perl', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_r_gpu() {
    return ['R: GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-r-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('gpu', mx_lib)
            utils.docker_run('ubuntu_gpu_cu101', 'unittest_ubuntu_gpu_R', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_julia07_cpu() {
    return ['Julia 0.7: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-it-julia07-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'unittest_ubuntu_cpu_julia07', false)
          }
        }
      }
    }]
}

def test_unix_julia10_cpu() {
    return ['Julia 1.0: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-it-julia10-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'unittest_ubuntu_cpu_julia10', false)
          }
        }
      }
    }]
}

def test_unix_onnx_cpu() {
    return ['Onnx CPU - Unit Tests': {
      node(NODE_LINUX_CPU) {
        ws('workspace/it-onnx-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'integrationtest_ubuntu_cpu_onnx', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_distributed_kvstore_cpu() {
    return ['dist-kvstore tests CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/it-dist-kvstore') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('cpu', mx_lib)
            utils.docker_run('ubuntu_cpu', 'integrationtest_ubuntu_cpu_dist_kvstore', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_distributed_kvstore_gpu() {
    return ['dist-kvstore tests GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/it-dist-kvstore') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('gpu', mx_lib)
            utils.docker_run('ubuntu_gpu_cu101', 'integrationtest_ubuntu_gpu_dist_kvstore', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_centos7_python3_cpu() {
    return ['Python3: CentOS 7 CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-centos7-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            try {
              utils.unpack_and_init('centos7_cpu', mx_lib)
              utils.docker_run('centos7_cpu', 'unittest_centos7_cpu', false)
              utils.publish_test_coverage()
            } finally {
              utils.collect_test_results_unix('nosetests_unittest.xml', 'nosetests_python3_centos7_cpu_unittest.xml')
              utils.collect_test_results_unix('nosetests_train.xml', 'nosetests_python3_centos7_cpu_train.xml')
            }
          }
        }
      }
    }]
}

def test_centos7_python3_gpu() {
    return ['Python3: CentOS 7 GPU': {
      node(NODE_LINUX_GPU) {
        ws('workspace/build-centos7-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            try {
              utils.unpack_and_init('centos7_gpu', mx_lib)
              utils.docker_run('centos7_gpu', 'unittest_centos7_gpu', true)
              utils.publish_test_coverage()
            } finally {
              utils.collect_test_results_unix('nosetests_gpu.xml', 'nosetests_python3_centos7_gpu.xml')
            }
          }
        }
      }
    }]
}

def test_centos7_scala_cpu() {
    return ['Scala: CentOS CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-scala-centos7-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('centos7_cpu', mx_lib)
            utils.docker_run('centos7_cpu', 'unittest_centos7_cpu_scala', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_windows_python3_gpu() {
    return ['Python 3: GPU Win':{
      node(NODE_WINDOWS_GPU) {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-gpu') {
            try {
              utils.init_git_win()
              unstash 'windows_package_gpu'
              powershell 'ci/windows/test_py3_gpu.ps1'
            } finally {
              utils.collect_test_results_windows('nosetests_forward.xml', 'nosetests_gpu_forward_windows_python3_gpu.xml')
              utils.collect_test_results_windows('nosetests_operator.xml', 'nosetests_gpu_operator_windows_python3_gpu.xml')
            }
          }
        }
      }
    }]
}

def test_windows_python3_gpu_mkldnn() {
    return ['Python 3: MKLDNN-GPU Win':{
      node(NODE_WINDOWS_GPU) {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-gpu') {
            try {
              utils.init_git_win()
              unstash 'windows_package_gpu_mkldnn'
              powershell 'ci/windows/test_py3_gpu.ps1'
            } finally {
              utils.collect_test_results_windows('nosetests_forward.xml', 'nosetests_gpu_forward_windows_python3_gpu_mkldnn.xml')
              utils.collect_test_results_windows('nosetests_operator.xml', 'nosetests_gpu_operator_windows_python3_gpu_mkldnn.xml')
            }
          }
        }
      }
    }]
}

def test_windows_python3_cpu() {
    return ['Python 3: CPU Win': {
      node(NODE_WINDOWS_CPU) {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-cpu') {
            try {
              utils.init_git_win()
              unstash 'windows_package_cpu'
              powershell 'ci/windows/test_py3_cpu.ps1'
            } finally {
              utils.collect_test_results_windows('nosetests_unittest.xml', 'nosetests_unittest_windows_python3_cpu.xml')
            }
          }
        }
      }
    }]
}

def test_windows_julia07_cpu() {
    return ['Julia 0.7: CPU Win': {
      node(NODE_WINDOWS_CPU) {
        ws('workspace/ut-julia07-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            unstash 'windows_package_cpu'
            powershell 'ci/windows/test_jl07_cpu.ps1'
          }
        }
      }
    }]
}

def test_windows_julia10_cpu() {
    return ['Julia 1.0: CPU Win': {
      node(NODE_WINDOWS_CPU) {
        ws('workspace/ut-julia10-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            unstash 'windows_package_cpu'
            powershell 'ci/windows/test_jl10_cpu.ps1'
          }
        }
      }
    }]
}

def test_qemu_armv7_cpu() {
    return ['ARMv7 QEMU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-armv7-qemu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('armv7', mx_cmake_lib)
            utils.docker_run('test.armv7', 'unittest_ubuntu_python3_arm', false)
          }
        }
      }
    }]
}

def test_qemu_armv8_cpu() {
    return ['ARMv8 QEMU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-armv8-qemu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('armv8', mx_cmake_lib)
            utils.docker_run('test.armv8', 'unittest_ubuntu_python3_arm', false)
          }
        }
      }
    }]
}

// This is for running on PRs
def docs_website() {
    return ['Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {

            unstash 'jekyll-artifacts'
            unstash 'python-artifacts'
            utils.docker_run('ubuntu_cpu_jekyll', 'build_docs_small', false)

            master_url = utils.get_jenkins_master_url()
            if ( master_url == 'jenkins.mxnet-ci.amazon-ml.com') {
                // TODO: Make sure this scripts publish the website from the right folder
                sh "ci/other/ci_deploy_doc.sh ${env.BRANCH_NAME} ${env.BUILD_NUMBER}"
            } else {
                print "Skipping staging documentation publishing since we are not running in prod. Host: {$master_url}"
            }
          }
        }
      }
    }]
}


// This creates the MXNet binary needed for generating different docs sets
def compile_unix_lite() {
    return ['MXNet lib': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu_lite', 'build_ubuntu_cpu_docs', false)
            utils.pack_lib('libmxnet', 'lib/libmxnet.so', false)
          }
        }
      }
    }]
}


def should_pack_website() {
  if (env.BRANCH_NAME) {
    if (env.BRANCH_NAME == "master" || env.BRANCH_NAME.startsWith("new_")) {
      return true
    }
  } else {
    return true
  }
  return false
}

// Each of the docs_{lang} functions will build the docs...
// Stashing is only needed for master for website publishing or for testing "new_"

// Call this function from Jenkins to generate just the Python API microsite artifacts.
def docs_python() {
    return ['Python Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('libmxnet', mx_lib, false)
            utils.docker_run('ubuntu_cpu_python', 'build_python_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('python-artifacts', 'docs/_build/python-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}


// Call this function from Jenkins to generate just the C and C++ API microsite artifacts.
def docs_c() {
    return ['C Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('libmxnet', 'lib/libmxnet.so', false)
            utils.docker_run('ubuntu_cpu_c', 'build_c_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('c-artifacts', 'docs/_build/c-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}


// Call this function from Jenkins to generate just the Julia API microsite artifacts.
def docs_julia() {
    return ['Julia Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('libmxnet', mx_lib, false)
            utils.docker_run('ubuntu_cpu_julia', 'build_julia_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('julia-artifacts', 'docs/_build/julia-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}


// Call this function from Jenkins to generate just the R API PDF artifact.
def docs_r() {
    return ['R Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('libmxnet', mx_lib, false)
            utils.docker_run('ubuntu_cpu_r', 'build_r_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('r-artifacts', 'docs/_build/r-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}


// Call this function from Jenkins to generate just the Scala API microsite artifacts.
// It will also generate the Scala package.
def docs_scala() {
    return ['Scala Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('libmxnet', mx_lib, false)
            utils.docker_run('ubuntu_cpu_scala', 'build_scala_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('scala-artifacts', 'docs/_build/scala-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}


// Call this function from Jenkins to generate just the Java API microsite artifacts.
// It will also generate the Scala package.
def docs_java() {
    return ['Java Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('libmxnet', mx_lib, false)
            utils.docker_run('ubuntu_cpu_scala', 'build_java_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('java-artifacts', 'docs/_build/java-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}


// Call this function from Jenkins to generate just the Clojure API microsite artifacts.
// It will also generate the Scala package.
def docs_clojure() {
    return ['Clojure Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init('libmxnet', mx_lib, false)
            utils.docker_run('ubuntu_cpu_scala', 'build_clojure_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('clojure-artifacts', 'docs/_build/clojure-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}


// Call this function from Jenkins to generate just the main website artifacts.
def docs_jekyll() {
    return ['Main Jekyll Website': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu_jekyll', 'build_jekyll_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('jekyll-artifacts', 'docs/_build/jekyll-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}

// This is for building the full website
// Assumes you have run all of the docs generation functions
def docs_full_website() {
    return ['Build artifacts full_website.tgz': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()

            unstash 'jekyll-artifacts'
            unstash 'c-artifacts'
            unstash 'python-artifacts'
            unstash 'r-artifacts'
            unstash 'julia-artifacts'
            unstash 'scala-artifacts'
            unstash 'java-artifacts'
            unstash 'clojure-artifacts'

            utils.docker_run('ubuntu_cpu_jekyll', 'build_docs', false)
            utils.pack_lib('full_website', 'docs/_build/full_website.tgz', false)
          }
        }
      }
    }]
}

// This is for uploading website artifacts to S3 bucket
// Assumes you have run docs_full_website function
def docs_upload_s3() {
    return ['Upload artifacts to s3 bucket': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            if(env.FOLDER_NAME) {
              utils.unpack_and_init('full_website', 'docs/_build/full_website.tgz')

              utils.docker_run('ubuntu_cpu', "push_docs ${env.FOLDER_NAME}", false)

              archiveArtifacts 'docs/_build/versions.zip'
            } else {
              sh 'echo Can not find website version for release. Please specify env var FOLDER_NAME in Jenkins pipeline'
              sh 'exit 1'
            }

          }
        }
      }
    }]
}

// This is for publishing the full website
// Assumes you have run all of the docs generation functions
// Called from Jenkins_website_full and Jenkins_website_full_pr
def docs_prepare() {
    return ['Prepare for publication of the full website': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()

            unstash 'jekyll-artifacts'
            unstash 'c-artifacts'
            unstash 'python-artifacts'
            unstash 'r-artifacts'
            unstash 'julia-artifacts'
            unstash 'scala-artifacts'
            unstash 'java-artifacts'
            unstash 'clojure-artifacts'

            utils.docker_run('ubuntu_cpu_jekyll', 'build_docs', false)

            // only stash if we're going to unstash later
            // utils.pack_lib('full_website', 'docs/_build/full_website.tgz', false)

            // archive so the publish pipeline can access the artifact
            archiveArtifacts 'docs/_build/full_website.tgz'
          }
        }
      }
    }]
}


def docs_prepare_beta() {
    return ['Prepare for publication to the staging website': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()

            unstash 'jekyll-artifacts'
            unstash 'python-artifacts'

            utils.docker_run('ubuntu_cpu_jekyll', 'build_docs_beta', false)

            // archive so the publish pipeline can access the artifact
            archiveArtifacts 'docs/_build/beta_website.tgz'
          }
        }
      }
    }]
}


def docs_archive() {
    return ['Archive the full website': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            archiveArtifacts 'docs/_build/full_website.tgz'
          }
        }
      }
    }]
}


// This is for the full website
def docs_publish() {
    return ['Publish the full website': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            // If used stashed files, you can retrieve them here
            //unstash 'full_website'
            //sh 'tar -xzf docs/_build/full_website.tgz --directory .'
            try {
              build 'restricted-website-publish-master'
            }
            catch (Exception e) {
               println(e.getMessage())
            }
          }
        }
      }
    }]
}


// This is for the beta website
def docs_publish_beta() {
    return ['Publish the beta website to staging': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            try {
              build 'restricted-website-publish-master-beta'
            }
            catch (Exception e) {
               println(e.getMessage())
            }
          }
        }
      }
    }]
}


def misc_asan_cpu() {
    return ['CPU ASAN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu-asan') {
            utils.unpack_and_init('cpu_asan', mx_lib_cpp_examples_cpu)
            utils.docker_run('ubuntu_cpu', 'integrationtest_ubuntu_cpu_asan', false)
        }
      }
    }]
}

def sanity_lint() {
    return ['Lint': {
      node(NODE_LINUX_CPU) {
        ws('workspace/sanity-lint') {
          utils.init_git()
          utils.docker_run('ubuntu_cpu', 'sanity_check', false)
        }
      }
    }]
}

def sanity_rat_license() {
    return ['RAT License': {
      node(NODE_LINUX_CPU) {
        ws('workspace/sanity-rat') {
          utils.init_git()
          utils.docker_run('ubuntu_rat', 'nightly_test_rat_check', false)
        }
      }
    }]
}

def test_artifact_repository() {
    return ['Test Artifact Repository Client': {
      node(NODE_LINUX_CPU) {
        ws('workspace/artifact-repo-client') {
          utils.init_git()
          utils.docker_run('ubuntu_cpu', 'test_artifact_repository', false)
        }
      }
    }]
}

return this
