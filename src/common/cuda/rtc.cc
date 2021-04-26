/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "mxnet/base.h"

#if MXNET_USE_CUDA

#include <nvrtc.h>

#include <mutex>
#include <string>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <algorithm>

#include "rtc.h"
#include "rtc/half-inl.h"
#include "rtc/util-inl.h"
#include "rtc/forward_functions-inl.h"
#include "rtc/backward_functions-inl.h"
#include "rtc/vectorization-inl.h"
#include "rtc/special_functions-inl.h"
#include "rtc/reducer-inl.h"
#include "utils.h"


namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

std::mutex lock;

namespace util {

std::string to_string(OpReqType req) {
  switch (req) {
    case kNullOp:
      return "OpReqType::kNullOp";
    case kWriteTo:
    case kWriteInplace:
      return "OpReqType::kWriteTo";
    case kAddTo:
      return "OpReqType::kAddTo";
  }
  LOG(FATAL) << "Unrecognized req.";
  return "";
}

}  // namespace util

namespace {

// Obtain compilation log from the program.
std::string GetCompileLog(nvrtcProgram program) {
  size_t log_size_including_null;
  NVRTC_CALL(nvrtcGetProgramLogSize(program, &log_size_including_null));
  std::string log(log_size_including_null - 1, '\0');
  // Room for terminating null character ensured since C++11
  NVRTC_CALL(nvrtcGetProgramLog(program, &log[0]));
  return log;
}

// Obtain compilation result (ptx assembly) from the program.
std::string GetCompiledCode(nvrtcProgram program, bool use_cubin) {
#if CUDA_VERSION >= 11010
  const auto getSize = use_cubin ? nvrtcGetCUBINSize : nvrtcGetPTXSize;
  const auto getFunc = use_cubin ? nvrtcGetCUBIN : nvrtcGetPTX;
#else
  const auto getSize = nvrtcGetPTXSize;
  const auto getFunc = nvrtcGetPTX;
#endif
  size_t ptx_size_including_null;
  NVRTC_CALL(getSize(program, &ptx_size_including_null));
  std::string ptx(ptx_size_including_null - 1, '\0');
  // Room for terminating null character ensured since C++11
  NVRTC_CALL(getFunc(program, &ptx[0]));
  return ptx;
}

std::tuple<bool, std::string> GetArchString(const int sm_arch) {
#if CUDA_VERSION < 10000
  constexpr int max_supported_sm_arch = 72;
#elif CUDA_VERSION < 11000
  constexpr int max_supported_sm_arch = 75;
#elif CUDA_VERSION < 11010
  constexpr int max_supported_sm_arch = 80;
#else
  constexpr int max_supported_sm_arch = 86;
#endif

#if CUDA_VERSION <= 11000
  // Always use PTX for CUDA <= 11.0
  const bool known_arch = false;
#else
  const bool known_arch = sm_arch <= max_supported_sm_arch;
#endif
  const int actual_sm_arch = std::min(sm_arch, max_supported_sm_arch);
  if (known_arch) {
    return {known_arch, "sm_" + std::to_string(actual_sm_arch)};
  } else {
    return {known_arch, "compute_" + std::to_string(actual_sm_arch)};
  }
}

}  // namespace

CUfunction get_function(const std::string &parameters,
                        const std::string &kernel_name,
                        const std::string &code,
                        int dev_id) {
  constexpr int CACHESIZE_WARN_THRESHOLD = 10000;
  std::lock_guard<std::mutex> l(lock);
  // Local class for value type of compile cache
  struct KernelInfo {
    std::string mangled_name;
    std::string ptx;
    std::vector<CUfunction> functions;
  };
  // Maps from the kernel name and parameters to the ptx and jit-compiled CUfunctions.
  using KernelCache = std::unordered_map<std::string, KernelInfo>;
  // Per-gpu-architecture compiled kernel cache with jit-compiled function for each device context
  static std::unordered_map<int32_t, KernelCache> compiled_kernels;
  int sm_arch = SMArch(dev_id);
  // make null map as needed
  KernelCache& compiled_kernels_this_arch = compiled_kernels[sm_arch];
  // make KernelInfo as needed
  KernelInfo& kinfo = compiled_kernels_this_arch[parameters + kernel_name];
  if (kinfo.ptx.size() == 0) {
    // It's the first time we've seen this kernel, so we need to generate the ptx and mangled_name.
    static std::string common_header =
        std::string(fp16_support_string) + "\n" +
        type_support_string + "\n" +
        util_string + "\n" +
        limits + "\n" +
        special_functions_definitions + '\n' +
        vectorization_support_string + "\n" +
        function_definitions_util + "\n" +
        function_definitions_binary + "\n" +
        function_definitions_unary + "\n" +
        backward_function_definitions + "\n" +
        grad_function_definitions + "\n" +
        reducer + "\n" +
        logic_reducer + "\n";
    std::string code_with_header = common_header + parameters + code;
    // If verbose mode, output kernel source, though not including the common header
    if (dmlc::GetEnv("MXNET_RTC_VERBOSE", false)) {
      LOG(INFO) << "\n" << std::string(80, '-') << "\n" << (parameters + code);
    }
    if (compiled_kernels_this_arch.size() == CACHESIZE_WARN_THRESHOLD + 1 &&
        dmlc::GetEnv("MXNET_RTC_SIZE_WARNING", true)) {
      LOG(WARNING) << "The number of different compiled kernels exceeds "
                   << CACHESIZE_WARN_THRESHOLD
                   << ".  Set MXNET_RTC_SIZE_WARNING=0 to quiet this warning.";
    }
    nvrtcProgram program;
    NVRTC_CALL(nvrtcCreateProgram(&program,                                  // prog
                                  &code_with_header[0],                      // buffer
                                  (kernel_name + "_kernel.cu").c_str(),      // name
                                  0,                                         // num headers
                                  nullptr,                                   // headers
                                  nullptr));                                 // include names
    const auto [use_cubin, gpu_arch] = GetArchString(sm_arch);  // NOLINT(*)
    std::string gpu_arch_arg = "--gpu-architecture=" + gpu_arch;
    const char *opts[] = {gpu_arch_arg.c_str(),
#if NDEBUG == 0
                          "-G",
#endif
                          "--std=c++14"};
    const std::string& kernel_name_demangled = kernel_name;
    NVRTC_CALL(nvrtcAddNameExpression(program, (kernel_name_demangled).c_str()));

    nvrtcResult compileResult = nvrtcCompileProgram(program,                         // prog
                                                    sizeof(opts) / sizeof(opts[0]),  // num options
                                                    opts);                           // options
    static const std::string dump_file = "mxnet_rtc_debug_code.log";
    if (compileResult != NVRTC_SUCCESS) {
      std::ofstream f(dump_file);
      f << code_with_header;
      f.close();
    }
    CHECK_EQ(compileResult, NVRTC_SUCCESS)
        << "NVRTC Compilation failed.\n"
        << "The generated code was stored in " << dump_file << "\n"
        << GetCompileLog(program);

    kinfo.ptx = GetCompiledCode(program, use_cubin);
    const char *mangled_name;
    NVRTC_CALL(nvrtcGetLoweredName(program,
                                   kernel_name_demangled.c_str(),
                                   &mangled_name));
    kinfo.mangled_name = mangled_name;
    // Destroy the program.
    NVRTC_CALL(nvrtcDestroyProgram(&program));
  }
  // Ensure function array is deep enough to index by dev_id
  while (kinfo.functions.size() <= static_cast<size_t>(dev_id))
    kinfo.functions.push_back(static_cast<CUfunction>(nullptr));
  // Jit-compile ptx for the device as needed
  if (kinfo.functions[dev_id] == static_cast<CUfunction>(nullptr)) {
    // Make sure driver context is set to the proper device
    CUdevice cu_device;
    CUcontext context;
    CUDA_DRIVER_CALL(cuDeviceGet(&cu_device, dev_id));
    CUDA_DRIVER_CALL(cuDevicePrimaryCtxRetain(&context, cu_device));
    // Jit-compile ptx for the driver's current context
    CUmodule module;

#if NDEBUG == 0
    intptr_t debug_info = 1;
    intptr_t line_info = 1;
#else
    intptr_t debug_info = 0;
    intptr_t line_info = 0;
#endif

    CUjit_option jit_opts[] = {CU_JIT_GENERATE_DEBUG_INFO, CU_JIT_GENERATE_LINE_INFO};
    void* jit_opt_values[] = {reinterpret_cast<void*>(debug_info),
                              reinterpret_cast<void*>(line_info)};

    CUDA_DRIVER_CALL(cuModuleLoadDataEx(&module, kinfo.ptx.c_str(), 2, jit_opts, jit_opt_values));
    CUDA_DRIVER_CALL(cuModuleGetFunction(&kinfo.functions[dev_id],
                                         module,
                                         kinfo.mangled_name.c_str()));
  }
  return kinfo.functions[dev_id];
}

void launch(CUfunction function,
            const dim3 grid_dim,
            const dim3 block_dim,
            unsigned int shared_mem_bytes,
            mshadow::Stream<gpu> *stream,
            std::vector<const void*> *args) {
  CHECK(args->size() != 0) <<
    "Empty argument list passed to a kernel.";
  // CUDA_DRIVER_CALL(
  CUresult err = cuLaunchKernel(function,                    // function to launch
    grid_dim.x, grid_dim.y, grid_dim.z,       // grid dim
    block_dim.x, block_dim.y, block_dim.z,    // block dim
    shared_mem_bytes,                         // shared memory
    mshadow::Stream<gpu>::GetStream(stream),  // stream
    const_cast<void**>(args->data()),         // arguments
    nullptr);  // );
  if (err != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(err, &error_string);
    LOG(FATAL) << "cuLaunchKernel failed: "
               << err << " " << error_string << ": "
               << reinterpret_cast<void*>(function) << " "
               << "(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ") "
               << "(" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ") "
               << shared_mem_bytes << " "
               << args->size();
  }
}

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
