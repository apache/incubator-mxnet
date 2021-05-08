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
mx_lib = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, example/extensions/lib_external_ops/build/libexternal_lib.so'
mx_lib_cython = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, example/extensions/lib_external_ops/build/libexternal_lib.so, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so'

// mxnet cmake libraries, in cmake builds we do not produce a libnvvm static library by default.
mx_cmake_lib = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/tests/mxnet_unit_tests'
mx_cmake_lib_no_tvm_op = 'build/libmxnet.so, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, example/extensions/lib_external_ops/build/libexternal_lib.so, build/tests/mxnet_unit_tests'
mx_cmake_lib_cython = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/tests/mxnet_unit_tests, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so'
// mxnet cmake libraries, in cmake builds we do not produce a libnvvm static library by default.
mx_cmake_lib_debug = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, build/tests/mxnet_unit_tests'
mx_onednn_lib = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, example/extensions/lib_external_ops/build/libexternal_lib.so'
mx_tensorrt_lib = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, lib/libnvonnxparser_runtime.so.0, lib/libnvonnxparser.so.0, lib/libonnx_proto.so, lib/libonnx.so'
mx_lib_cpp_examples = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, example/extensions/lib_external_ops/build/libexternal_lib.so, build/cpp-package/example/*, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so'
mx_lib_cpp_examples_no_tvm_op = 'build/libmxnet.so, build/libcustomop_lib.so, build/libcustomop_gpu_lib.so, build/libsubgraph_lib.so, python/mxnet/_cy3/*.so, python/mxnet/_ffi/_cy3/*.so'
mx_lib_cpp_examples_cpu = 'build/libmxnet.so, build/3rdparty/tvm/libtvm_runtime.so, build/libtvmop.so, build/tvmop.conf, build/cpp-package/example/*'
mx_cd_lib = 'lib/libmxnet.so, licenses/*, lib/libgfortran.so.*, lib/libopenblas.so.0, include/onednn/oneapi/dnnl/dnnl_version.h, include/onednn/oneapi/dnnl/dnnl_config.h'


// Python unittest for CPU
// Python 3
def python3_ut(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    utils.docker_run(docker_container_name, 'unittest_ubuntu_python3_cpu', false)
  }
}

def python3_ut_onednn(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    utils.docker_run(docker_container_name, 'unittest_ubuntu_python3_cpu_onednn', false)
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

def compile_unix_cpu_openblas(lib_name) {
    return ['CPU: Openblas': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-openblas') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_openblas', false)
            utils.pack_lib(lib_name, mx_lib_cpp_examples, true)
          }
        }
      }
    }]
}

def compile_unix_openblas_debug_cpu(lib_name) {
    return ['CPU: Openblas, cmake, debug': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-openblas') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_cmake_debug', false)
            utils.pack_lib(lib_name, mx_cmake_lib_debug, true)
          }
        }
      }
    }]
}

def compile_unix_openblas_cpu_no_tvm_op(lib_name) {
    return ['CPU: Openblas, cmake, TVM_OP OFF': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-openblas-no-tvm-op') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_cmake_no_tvm_op', false)
            utils.pack_lib(lib_name, mx_cmake_lib_no_tvm_op)
          }
        }
      }
    }]
}

def compile_unix_int64_cpu(lib_name) {
    return ['CPU: USE_INT64_TENSOR_SIZE': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-int64') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run(lib_name, 'build_ubuntu_cpu_large_tensor', false)
          }
        }
      }
    }]
}

def compile_unix_int64_gpu(lib_name) {
    return ['GPU: USE_INT64_TENSOR_SIZE': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/build-gpu-int64') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_cu111', 'build_ubuntu_gpu_large_tensor', false)
            utils.pack_lib(lib_name, mx_cmake_lib)
          }
        }
      }
    }]
}

def compile_unix_mkl_cpu(lib_name) {
    return ['CPU: MKL': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-mkl') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_mkl', false)
            utils.pack_lib(lib_name, mx_lib, false)
          }
        }
      }
    }]
}

def compile_unix_onednn_cpu(lib_name) {
    return ['CPU: ONEDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-onednn-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_onednn', false)
            utils.pack_lib(lib_name, mx_onednn_lib, true)
          }
        }
      }
    }]
}

def compile_unix_onednn_mkl_cpu(lib_name) {
    return ['CPU: ONEDNN_MKL': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-onednn-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_onednn_mkl', false)
            utils.pack_lib(lib_name, mx_onednn_lib, false)
          }
        }
      }
    }]
}

def compile_unix_onednn_gpu(lib_name) {
    return ['GPU: ONEDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-onednn-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_cu111', 'build_ubuntu_gpu_onednn', false)
            utils.pack_lib(lib_name, mx_onednn_lib)
          }
        }
      }
    }]
}

def compile_unix_onednn_nocudnn_gpu(lib_name) {
    return ['GPU: ONEDNN_CUDNNOFF': {
       node(NODE_LINUX_CPU) {
         ws('workspace/build-onednn-gpu-nocudnn') {
           timeout(time: max_time, unit: 'MINUTES') {
             utils.init_git()
             utils.docker_run('ubuntu_gpu_cu111', 'build_ubuntu_gpu_onednn_nocudnn', false)
             utils.pack_lib(lib_name, mx_onednn_lib)
           }
         }
       }
    }]
}

def compile_unix_full_gpu(lib_name) {
    return ['GPU: CUDA+cuDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_cu111', 'build_ubuntu_gpu', false)
            utils.pack_lib(lib_name, mx_lib_cpp_examples)
          }
        }
      }
    }]
}

def compile_unix_full_gpu_debug(lib_name) {
    return ['GPU: CUDA+cuDNN, debug': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_cu111', 'build_ubuntu_gpu_debug', false)
            utils.pack_lib(lib_name, mx_lib_cpp_examples)
          }
        }
      }
    }]
}

def compile_unix_tensorrt_gpu(lib_name) {
    return ['TensorRT': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-tensorrt') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_tensorrt_cu111', 'build_ubuntu_gpu_tensorrt', false)
            utils.pack_lib(lib_name, mx_tensorrt_lib)
          }
        }
      }
    }]
}

def compile_centos7_cpu(lib_name) {
    return ['CPU: CentOS 7': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-centos7-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_cpu', 'build_centos7_cpu', false)
            utils.pack_lib(lib_name, mx_lib)
          }
        }
      }
    }]
}

def compile_centos7_cpu_onednn() {
    return ['CPU: CentOS 7 ONEDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-centos7-onednn') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_cpu', 'build_centos7_onednn', false)
          }
        }
      }
    }]
}

def compile_centos7_gpu(lib_name) {
    return ['GPU: CentOS 7': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-centos7-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_gpu_cu102', 'build_centos7_gpu', false)
            utils.pack_lib(lib_name, mx_lib)
          }
        }
      }
    }]
}

def compile_unix_clang_6_cpu() {
    return ['CPU: Clang 6': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-clang39') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang6', false)
          }
        }
      }
    }]
}

// TODO(leezu) delete once DUSE_DIST_KVSTORE=ON builds in -WError build
def compile_unix_clang_10_cpu() {
    return ['CPU: Clang 10': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-clang100') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang100', false)
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

def compile_unix_clang_6_onednn_cpu() {
    return ['CPU: Clang 6 ONEDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-onednn-clang6') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang6_onednn', false)
          }
        }
      }
    }]
}

// TODO(leezu) delete once DUSE_DIST_KVSTORE=ON builds in -WError build
def compile_unix_clang_10_onednn_cpu() {
    return ['CPU: Clang 10 ONEDNN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-onednn-clang100') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang100_onednn', false)
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

def compile_armv6_cpu(lib_name) {
    return ['ARMv6':{
      node(NODE_LINUX_CPU) {
        ws('workspace/build-ARMv6') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('armv6', 'build_armv6', false)
            utils.pack_lib(lib_name, mx_lib)
          }
        }
      }
    }]
}

def compile_armv7_cpu(lib_name) {
    return ['ARMv7':{
      node(NODE_LINUX_CPU) {
        ws('workspace/build-ARMv7') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('armv7', 'build_armv7', false)
            utils.pack_lib(lib_name, mx_lib)
          }
        }
      }
    }]
}

def compile_armv8_cpu(lib_name) {
    return ['ARMv8':{
      node(NODE_LINUX_CPU) {
        ws('workspace/build-ARMv8') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('armv8', 'build_armv8', false)
            utils.pack_lib(lib_name, mx_lib)
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

def compile_unix_asan_cpu(lib_name) {
    return ['CPU: ASAN': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-asan') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_cmake_asan', false)
            utils.pack_lib(lib_name, mx_lib_cpp_examples_cpu)
          }
        }
      }
    }]
}

def compile_unix_gcc8_werror(lib_name) {
    return ['CPU: GCC8 -WError': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-gcc8') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_gcc8_werror', false)
            utils.pack_lib(lib_name, mx_lib)
          }
        }
      }
    }]
}

def compile_unix_clang10_werror(lib_name) {
    return ['CPU: Clang10 -WError': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-clang10') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_cpu', 'build_ubuntu_cpu_clang10_werror', false)
            utils.pack_lib(lib_name, mx_lib)
          }
        }
      }
    }]
}

def compile_unix_clang10_cuda_werror(lib_name) {
    return ['GPU: Clang10 -WError': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-cpu-clang10') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('ubuntu_gpu_cu111', 'build_ubuntu_gpu_clang10_werror', false)
            utils.pack_lib(lib_name, mx_lib)
          }
        }
      }
    }]
}

def compile_windows_cpu(lib_name) {
    return ['Build CPU windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_CPU'
            stash includes: 'windows_package.7z', name: lib_name
          }
        }
      }
    }]
}

def compile_windows_cpu_onednn(lib_name) {
    return ['Build CPU ONEDNN windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-cpu-onednn') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_CPU_ONEDNN'
            stash includes: 'windows_package.7z', name: lib_name
          }
        }
      }
    }]
}

def compile_windows_cpu_onednn_mkl(lib_name) {
    return ['Build CPU ONEDNN MKL windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-cpu-onednn-mkl') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_CPU_ONEDNN_MKL'
            stash includes: 'windows_package.7z', name: lib_name
          }
        }
      }
    }]
}

def compile_windows_cpu_mkl(lib_name) {
    return ['Build CPU MKL windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-cpu-mkl') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_CPU_MKL'
            stash includes: 'windows_package.7z', name: lib_name
          }
        }
      }
    }]
}

def compile_windows_gpu(lib_name) {
    return ['Build GPU windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
              utils.init_git_win()
              powershell 'py -3 ci/build_windows.py -f WIN_GPU'
              stash includes: 'windows_package.7z', name: lib_name
          }
        }
      }
    }]
}

def compile_windows_gpu_onednn(lib_name) {
    return ['Build GPU ONEDNN windows':{
      node(NODE_WINDOWS_CPU) {
        ws('workspace/build-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git_win()
            powershell 'py -3 ci/build_windows.py -f WIN_GPU_ONEDNN'
            stash includes: 'windows_package.7z', name: lib_name
          }
        }
      }
    }]
}

def compile_static_python_cpu() {
  return ['Static build CPU CentOS7 Python' : {
    node(NODE_LINUX_CPU) {
        ws('workspace/ut-publish-python-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_cpu', 'build_static_python_cpu', false)
          }
        }
    }
  }]
}

def compile_static_cd_cpu(lib_name) {
  return ['CPU: CD Static Build' : {
    node(NODE_LINUX_CPU) {
        ws('workspace/build-cd-static/cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_cpu', 'build_static_libmxnet cpu', false)
            utils.pack_lib(lib_name, mx_cd_lib)
          }
        }
    }
  }]
}

def compile_static_python_gpu() {
  return ['Static build GPU CentOS7 Python' : {
    node(NODE_LINUX_GPU) {
        ws('workspace/ut-publish-python-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_gpu_cu102', 'build_static_python_cu102')
          }
        }
    }
  }]
}

def compile_static_cd_gpu(lib_name) {
  return ['GPU: CD Static Build' : {
    node(NODE_LINUX_CPU) {
        ws('workspace/build-cd-static/gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()
            utils.docker_run('centos7_gpu_cu102', 'build_static_libmxnet cu102', false)
            utils.pack_lib(lib_name, mx_cd_lib)
          }
        }
    }
  }]
}

def test_unix_python3_cpu(lib_name) {
    return ['Python3: CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu') {
          try {
            utils.unpack_and_init(lib_name, mx_lib, true)
            python3_ut('ubuntu_cpu')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('tests_unittest.xml', 'tests_python3_cpu_unittest.xml')
            utils.collect_test_results_unix('tests_quantization.xml', 'tests_python3_cpu_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_mkl_cpu(lib_name) {
    return ['Python3: MKL-CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu') {
          try {
            utils.unpack_and_init(lib_name, mx_lib)
            python3_ut('ubuntu_cpu')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('tests_unittest.xml', 'tests_python3_cpu_unittest.xml')
            utils.collect_test_results_unix('tests_quantization.xml', 'tests_python3_cpu_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_gpu(lib_name) {
    return ['Python3: GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-gpu') {
          try {
            utils.unpack_and_init(lib_name, mx_lib_cython)
            python3_gpu_ut_cython('ubuntu_gpu_cu111')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('tests_gpu.xml', 'tests_python3_gpu.xml')
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
            utils.unpack_and_init('cpu_debug', mx_cmake_lib_debug, true)
            python3_ut('ubuntu_cpu')
          } finally {
            utils.collect_test_results_unix('tests_unittest.xml', 'tests_python3_cpu_debug_unittest.xml')
            utils.collect_test_results_unix('tests_quantization.xml', 'tests_python3_cpu_debug_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_cpu_no_tvm_op(lib_name) {
    return ['Python3: CPU TVM_OP OFF': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-cpu-no-tvm-op') {
          try {
            utils.unpack_and_init(lib_name, mx_cmake_lib_no_tvm_op)
            python3_ut('ubuntu_cpu')
          } finally {
            utils.collect_test_results_unix('tests_unittest.xml', 'tests_python3_cpu_no_tvm_op_unittest.xml')
            utils.collect_test_results_unix('tests_quantization.xml', 'tests_python3_cpu_no_tvm_op_quantization.xml')
          }
        }
      }
    }]
}

def test_unix_python3_onednn_cpu(lib_name) {
    return ['Python3: ONEDNN-CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-onednn-cpu') {
          try {
            utils.unpack_and_init(lib_name, mx_onednn_lib, true)
            python3_ut_onednn('ubuntu_cpu')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('tests_unittest.xml', 'tests_python3_onednn_cpu_unittest.xml')
            utils.collect_test_results_unix('tests_mkl.xml', 'tests_python3_onednn_cpu_mkl.xml')
          }
        }
      }
    }]
}

def test_unix_python3_onednn_mkl_cpu(lib_name) {
    return ['Python3: ONEDNN-MKL-CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-python3-onednn-mkl-cpu') {
          try {
            utils.unpack_and_init(lib_name, mx_lib)
            python3_ut_onednn('ubuntu_cpu')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('tests_unittest.xml', 'tests_python3_onednn_cpu_unittest.xml')
            utils.collect_test_results_unix('tests_mkl.xml', 'tests_python3_onednn_cpu_mkl.xml')
          }
        }
      }
    }]
}

def test_unix_python3_onednn_gpu(lib_name) {
    return ['Python3: ONEDNN-GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-onednn-gpu') {
          try {
            utils.unpack_and_init(lib_name, mx_onednn_lib)
            python3_gpu_ut('ubuntu_gpu_cu111')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('tests_gpu.xml', 'tests_python3_onednn_gpu.xml')
          }
        }
      }
    }]
}

def test_unix_python3_onednn_nocudnn_gpu(lib_name) {
    return ['Python3: ONEDNN-GPU-NOCUDNN': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/ut-python3-onednn-gpu-nocudnn') {
          try {
            utils.unpack_and_init(lib_name, mx_onednn_lib)
            python3_gpu_ut_nocudnn('ubuntu_gpu_cu111')
            utils.publish_test_coverage()
          } finally {
            utils.collect_test_results_unix('tests_gpu.xml', 'tests_python3_onednn_gpu_nocudnn.xml')
          }
        }
      }
    }]
}

def test_unix_onnx_cpu(lib_name) {
    return ['Onnx: CPU Makefile': {
      node(NODE_LINUX_CPU) {
        ws('workspace/it-onnx-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib)
            utils.docker_run('ubuntu_cpu', 'integrationtest_ubuntu_cpu_onnx', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_distributed_kvstore_cpu(lib_name) {
    return ['dist-kvstore tests CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/it-dist-kvstore') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib, true)
            utils.docker_run('ubuntu_cpu', 'integrationtest_ubuntu_cpu_dist_kvstore', false)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_byteps_gpu(lib_name) {
    return ['byteps tests GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/it-byteps') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib)
            utils.docker_run('ubuntu_gpu_cu111', 'integrationtest_ubuntu_gpu_byteps', true, '32768m')
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_distributed_kvstore_gpu(lib_name) {
    return ['dist-kvstore tests GPU': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/it-dist-kvstore') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib)
            utils.docker_run('ubuntu_gpu_cu111', 'integrationtest_ubuntu_gpu_dist_kvstore', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_unix_cpp_package_gpu(lib_name) {
    return ['cpp-package GPU Makefile': {
      node(NODE_LINUX_GPU_G4) {
        ws('workspace/it-cpp-package-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib_cpp_examples)
            utils.docker_run('ubuntu_gpu_cu111', 'integrationtest_ubuntu_cpp_package_gpu', true)
            utils.publish_test_coverage()
          }
        }
      }
    }]
}

def test_centos7_python3_cpu(lib_name) {
    return ['Python3: CentOS 7 CPU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/build-centos7-cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            try {
              utils.unpack_and_init(lib_name, mx_lib)
              utils.docker_run('centos7_cpu', 'unittest_centos7_cpu', false)
              utils.publish_test_coverage()
            } finally {
              utils.collect_test_results_unix('tests_unittest.xml', 'tests_python3_centos7_cpu_unittest.xml')
              utils.collect_test_results_unix('tests_train.xml', 'tests_python3_centos7_cpu_train.xml')
            }
          }
        }
      }
    }]
}

def test_centos7_python3_cd_cpu(lib_name) {
    return ['Python3: CentOS 7 CPU CD': {
      node(NODE_LINUX_CPU) {
        ws('workspace/test-cd-static/cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_cd_lib)
            utils.docker_run('centos7_cpu', 'cd_unittest_ubuntu cpu', false)
          }
        }
      }
    }]
}

def test_centos7_pypi_package_cd_cpu(lib_name) {
    return ['PyPI package: CentOS 7 CPU CD': {
      node(NODE_LINUX_CPU) {
        ws('workspace/test-cd-pypi/cpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_cd_lib)
            utils.docker_run('centos7_cpu', 'ci_package_pypi cpu', false)
          }
        }
      }
    }]
}

def test_centos7_python3_gpu(lib_name) {
    return ['Python3: CentOS 7 GPU': {
      node(NODE_LINUX_GPU) {
        ws('workspace/build-centos7-gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            try {
              utils.unpack_and_init(lib_name, mx_lib)
              utils.docker_run('centos7_gpu_cu102', 'unittest_centos7_gpu', true)
              utils.publish_test_coverage()
            } finally {
              utils.collect_test_results_unix('tests_gpu.xml', 'tests_python3_centos7_gpu.xml')
            }
          }
        }
      }
    }]
}

def test_centos7_python3_cd_gpu(lib_name) {
    return ['Python3: CentOS 7 GPU CD': {
      node(NODE_LINUX_GPU) {
        ws('workspace/test-cd-static/gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_cd_lib)
            utils.docker_run('centos7_gpu_cu102', 'cd_unittest_ubuntu cu102', true)
          }
        }
      }
    }]
}

def test_centos7_pypi_package_cd_gpu(lib_name) {
    return ['PyPI package: CentOS 7 GPU CD': {
      node(NODE_LINUX_GPU) {
        ws('workspace/test-cd-pypi/gpu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_cd_lib)
            utils.docker_run('centos7_gpu_cu102', 'ci_package_pypi cu102', true)
          }
        }
      }
    }]
}

def test_windows_python3_gpu(lib_name) {
    return ['Python 3: GPU Win':{
      node(NODE_WINDOWS_GPU) {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-gpu') {
            try {
              utils.init_git_win()
              unstash lib_name
              powershell 'ci/windows/test_py3_gpu.ps1'
            } finally {
              utils.collect_test_results_windows('tests_forward.xml', 'tests_gpu_forward_windows_python3_gpu.xml')
              utils.collect_test_results_windows('tests_operator.xml', 'tests_gpu_operator_windows_python3_gpu.xml')
            }
          }
        }
      }
    }]
}

def test_windows_python3_gpu_onednn(lib_name) {
    return ['Python 3: ONEDNN-GPU Win':{
      node(NODE_WINDOWS_GPU) {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-gpu') {
            try {
              utils.init_git_win()
              unstash lib_name
              powershell 'ci/windows/test_py3_gpu.ps1'
            } finally {
              utils.collect_test_results_windows('tests_forward.xml', 'tests_gpu_forward_windows_python3_gpu_onednn.xml')
              utils.collect_test_results_windows('tests_operator.xml', 'tests_gpu_operator_windows_python3_gpu_onednn.xml')
            }
          }
        }
      }
    }]
}

def test_windows_python3_cpu(lib_name) {
    return ['Python 3: CPU Win': {
      node(NODE_WINDOWS_CPU) {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-cpu') {
            try {
              utils.init_git_win()
              unstash lib_name
              powershell 'ci/windows/test_py3_cpu.ps1'
            } finally {
              utils.collect_test_results_windows('tests_unittest.xml', 'tests_unittest_windows_python3_cpu.xml')
            }
          }
        }
      }
    }]
}

def test_qemu_armv7_cpu(lib_name) {
    return ['ARMv7 QEMU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-armv7-qemu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib)
            utils.docker_run('test.armv7', 'unittest_ubuntu_python3_arm', false)
          }
        }
      }
    }]
}

def test_qemu_armv8_cpu(lib_name) {
    return ['ARMv8 QEMU': {
      node(NODE_LINUX_CPU) {
        ws('workspace/ut-armv8-qemu') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib)
            utils.docker_run('test.armv8', 'unittest_ubuntu_python3_arm', false)
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
def docs_python(lib_name) {
    return ['Python Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib, false)
            utils.docker_run('ubuntu_cpu', 'build_python_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('python-artifacts', 'docs/_build/python-artifacts.tgz', false)
            }
          }
        }
      }
    }]
}

// Call this function from Jenkins to generate just the C and C++ API microsite artifacts.
def docs_c(lib_name) {
    return ['C Docs': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.unpack_and_init(lib_name, mx_lib, false)
            utils.docker_run('ubuntu_cpu', 'build_c_docs', false)
            if (should_pack_website()) {
              utils.pack_lib('c-artifacts', 'docs/_build/c-artifacts.tgz', false)
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

// This is for updateing the new version of website artifact
// Assumes you have run all of the docs generation functions
// Called from Jenkins_website_version_artifacts
def docs_full_website() {
    return ['Build artifacts full_website.tgz': {
      node(NODE_LINUX_CPU) {
        ws('workspace/docs') {
          timeout(time: max_time, unit: 'MINUTES') {
            utils.init_git()

            unstash 'jekyll-artifacts'
            unstash 'c-artifacts'
            unstash 'python-artifacts'

            utils.docker_run('ubuntu_cpu_jekyll', 'build_docs', false)

            utils.pack_lib('full_website', 'docs/_build/full_website.tgz', false)

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
            unstash 'c-artifacts'
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
          utils.docker_run('ubuntu_cpu', 'test_rat_check', false)
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

def misc_test_docker_cache_build() {
  return ['Test Docker cache build': {
    node(NODE_LINUX_CPU) {
      ws('workspace/docker_cache') {
        utils.init_git()
        sh "cd ci && docker-compose -f docker/docker-compose.yml pull && docker-compose -f docker/docker-compose.yml build --parallel"
      }
    }
  }]
}

return this
