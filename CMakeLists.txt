cmake_minimum_required(VERSION 3.13)

# workaround to store CMAKE_CROSSCOMPILING because is getting reset by the project command
if(CMAKE_CROSSCOMPILING)
  set(__CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING})
  set(__CMAKE_CROSSCOMPILING_OVERRIDE ON)
endif()

project(mxnet C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # GNU extensions used by src/operator/random/shuffle_op.cc

# Sanity checks for some popular compilers. Make sure their version is
# sufficient. Cmake also automatically checks if a compiler supports c++17. But
# some compilers claim they support c++17 without actually implementing crucial
# parts of the standard leading to hard to understand compilation errors.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "MXNet 2 requires a C++17 compatible compiler. Please update to GCC version 7 or newer.")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    message(FATAL_ERROR "MXNet 2 requires a C++17 compatible compiler. Please update to Clang version 6 or newer.")
  endif()
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  # Load config.cmake only if mxnet is not compiled as a dependency of another project
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
endif()

if(__CMAKE_CROSSCOMPILING_OVERRIDE)
  set(CMAKE_CROSSCOMPILING ${__CMAKE_CROSSCOMPILING})
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Utils.cmake)

include(CMakeDependentOption)
#Some things have order. This must be put in front alone
option(MXNET_BUILD_SHARED_LIBS "Build shared libraries instead of static libraries" ON)
option(USE_CUDA "Build with CUDA support"   ON)
set(MXNET_CUDA_ARCH "Auto" CACHE STRING "Target NVIDIA GPU achitecture.
Format: Auto | Common | All | LIST(ARCH_AND_PTX ...)
- \"Auto\" detects local machine GPU compute arch at runtime.
- \"Common\" and \"All\" cover common and entire subsets of architectures
- ARCH_AND_PTX : NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
- NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal Volta Turing
- NUM: Any number. Only those pairs are currently accepted by NVCC though:
       2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2 7.0 7.2 7.5")
option(USE_NCCL "Use NVidia NCCL with CUDA" OFF)
option(USE_OPENCV "Build with OpenCV support" ON)
option(USE_OPENMP "Build with Openmp support" ON)
option(USE_FATBIN_COMPRESSION "Compress nvcc fatbin output" ON)
cmake_dependent_option(USE_NVML "Build with nvml support if found" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_CUDNN "Build with cudnn support" ON "USE_CUDA" OFF) # one could set CUDNN_ROOT for search path
cmake_dependent_option(USE_CUTENSOR "Build with cuTENSOR support" ON "USE_CUDA" OFF) # one could set CUTENSOR_ROOT for search path
cmake_dependent_option(USE_NVTX "Build with nvtx support if found" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_SSE "Build with x86 SSE instruction support" ON
  "CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL amd64" OFF)
option(USE_F16C "Build with x86 F16C instruction support" ON) # autodetects support if ON
option(USE_LAPACK "Build with lapack support" ON)
option(USE_MKL_LAYERNORM "Use layer normalization from MKL, which is currently slower than internal. No effect unless USE_BLAS=MKL (or mkl)." OFF)
if((NOT APPLE) AND (NOT MSVC) AND (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64") AND (NOT CMAKE_CROSSCOMPILING))
  option(USE_ONEDNN "Build with ONEDNN support" ON)
else()
  option(USE_ONEDNN "Build with ONEDNN support" OFF)
endif()
cmake_dependent_option(USE_INTGEMM "Build with x86_64 intgemm library for low-precision multiplication" ON "CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64" OFF)
if(NOT MSVC)
  option(USE_OPERATOR_TUNING  "Enable auto-tuning of operators" ON)
else()
  option(USE_OPERATOR_TUNING  "Enable auto-tuning of operators" OFF)
endif()
option(USE_GPERFTOOLS "Build with GPerfTools support" OFF)
option(USE_JEMALLOC "Build with Jemalloc support" OFF)
option(USE_LIBJPEG_TURBO "Use libjpeg-turbo" OFF)
option(USE_DIST_KVSTORE "Build with DIST_KVSTORE support" OFF)
option(USE_PLUGINS_WARPCTC "Use WARPCTC Plugins" OFF)
option(USE_CPP_PACKAGE "Build C++ Package" OFF)
option(USE_MXNET_LIB_NAMING "Use MXNet library naming conventions." ON)
option(USE_GPROF "Compile with gprof (profiling) flag" OFF)
option(USE_VTUNE "Enable use of Intel Amplifier XE (VTune)" OFF) # one could set VTUNE_ROOT for search path
option(USE_TVM_OP "Enable use of TVM operator build system." OFF)
option(BUILD_CPP_EXAMPLES "Build cpp examples" ON)
option(INSTALL_EXAMPLES "Install the example source files." OFF)
option(USE_SIGNAL_HANDLER "Print stack traces on segfaults." ON)
option(USE_TENSORRT "Enable inference optimization with TensorRT." OFF)
option(USE_ASAN "Enable Clang/GCC ASAN sanitizers." OFF)
cmake_dependent_option(ENABLE_TESTCOVERAGE "Enable compilation with test coverage metric output" OFF "NOT MSVC" OFF)
option(BUILD_EXTENSION_PATH "Path to extension to build" "")
option(BUILD_CYTHON_MODULES "Build cython modules." OFF)
option(LOG_FATAL_THROW "Log exceptions but do not abort" ON)
cmake_dependent_option(USE_SPLIT_ARCH_DLL "Build a separate DLL for each Cuda arch (Windows only)." ON "MSVC" OFF)
cmake_dependent_option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON "UNIX" OFF)
cmake_dependent_option(MXNET_FORCE_SHARED_CRT "Build with dynamic CRT on Windows (/MD)" ON "MXNET_BUILD_SHARED_LIBS" OFF)

message(STATUS "CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING}")
message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")

message(STATUS "CMAKE_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}")

if(USE_TVM_OP)
  add_definitions(-DMXNET_USE_TVM_OP=1)
endif()

if(MXNET_FORCE_SHARED_CRT)
  set(DMLC_FORCE_SHARED_CRT ON)
  set(gtest_force_shared_crt ON)
endif()

message(STATUS "CMake version '${CMAKE_VERSION}' using generator '${CMAKE_GENERATOR}'")
project(mxnet C CXX)
if(USE_CUDA)
  cmake_minimum_required(VERSION 3.13.2)  # CUDA 10 (Turing) detection available starting 3.13.2
  include(CheckLanguage)
  check_language(CUDA)
  if (NOT CMAKE_CUDA_COMPILER AND UNIX AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(ENV{CUDACXX} "/usr/local/cuda/bin/nvcc")
    message(WARNING "CMAKE_CUDA_COMPILER guessed: " $ENV{CUDACXX} "\n"
      "Please fix your cuda installation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#mandatory-post")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(UNIX)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  else()
    message(STATUS "Could not find CCache. Consider installing CCache to speed up compilation.")
  endif()
endif()

if(MSVC)
  set(SYSTEM_ARCHITECTURE x86_64)
  enable_language(ASM_MASM)
else()
  execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE SYSTEM_ARCHITECTURE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Distribution")
  if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH $\{ORIGIN\})
    # Enforce DT_PATH instead of DT_RUNPATH
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")
  endif()
  set(Protobuf_USE_STATIC_LIBS ON)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/upstream;${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules;${CMAKE_MODULE_PATH}")

SET(EXTRA_OPERATORS "" CACHE PATH "EXTRA OPERATORS PATH")

if("$ENV{VERBOSE}" STREQUAL "1")
  message(STATUS " Verbose Makefile ACTIVATED")
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

#Switch off modern thread local for dmlc-core, please see: https://github.com/dmlc/dmlc-core/issues/571#issuecomment-543467484
add_definitions(-DDMLC_MODERN_THREAD_LOCAL=0)
# disable stack trace in exception by default.
add_definitions(-DDMLC_LOG_STACK_TRACE_SIZE=0)

add_definitions(-DDMLC_USE_CXX11)
add_definitions(-DDMLC_STRICT_CXX11)
add_definitions(-DDMLC_USE_CXX14)
add_definitions(-DMSHADOW_IN_CXX11)
if(MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DMXNET_EXPORTS)
  add_definitions(-DNNVM_EXPORTS)
  add_definitions(-DNOMINMAX)
  set(CMAKE_C_FLAGS "/MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS} /bigobj")
else()
  include(CheckCXXCompilerFlag)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-sign-compare")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_ASSERTIONS")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_ASSERTIONS")
  else()
    add_definitions(-DNDEBUG=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}")
endif()

if(NOT mxnet_LINKER_LIBS)
  set(mxnet_LINKER_LIBS "")
endif()

if(USE_GPROF)
  message(STATUS "Using GPROF")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_LINK_LIBRARY_FILE_FLAG "${CMAKE_LINK_LIBRARY_FILE_FLAG} -g -pg")
endif()

if(USE_VTUNE)
  message(STATUS "Using VTUNE")
  if(NOT VTUNE_ROOT)
    set(VTUNE_ROOT /opt/intel/vtune_amplifier_xe_2017)
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_LINK_LIBRARY_FILE_FLAG "${CMAKE_LINK_LIBRARY_FILE_FLAG} -g -pg")
  add_definitions(-DMXNET_USE_VTUNE=1)
  include_directories(${VTUNE_ROOT}/include)
  list(APPEND mxnet_LINKER_LIBS ${VTUNE_ROOT}/lib64/libittnotify.a)
  list(APPEND mxnet_LINKER_LIBS dl)
endif()

if(USE_TENSORRT)
  message(STATUS "Using TensorRT")
  set(ONNX_PATH 3rdparty/onnx-tensorrt/third_party/onnx/build/)
  set(ONNX_TRT_PATH 3rdparty/onnx-tensorrt/build/)

  include_directories(${ONNX_PATH})
  include_directories(3rdparty/onnx-tensorrt/)
  include_directories(3rdparty/)
  include_directories(3rdparty/onnx-tensorrt/third_party/onnx/)
  add_definitions(-DMXNET_USE_TENSORRT=1)
  add_definitions(-DONNX_NAMESPACE=onnx)
  add_definitions(-DONNX_ML=1)

  find_package(Protobuf REQUIRED)

  find_library(ONNX_LIBRARY NAMES libonnx.so REQUIRED
          PATHS ${ONNX_PATH}
          DOC "Path to onnx library.")
  find_library(ONNX_PROTO_LIBRARY NAMES libonnx_proto.so REQUIRED
          PATHS ${ONNX_PATH}
          DOC "Path to onnx_proto library.")
  find_library(ONNX_TRT_PARSER_LIBRARY NAMES libnvonnxparser.so REQUIRED
          PATHS ${ONNX_TRT_PATH}
          DOC "Path to onnx_proto parser library.")

  list(APPEND mxnet_LINKER_LIBS libnvinfer.so ${ONNX_TRT_PARSER_LIBRARY}
          ${ONNX_PROTO_LIBRARY} ${ONNX_LIBRARY} ${PROTOBUF_LIBRARY})
endif()

if(USE_MKL_LAYERNORM)
  add_definitions(-DMXNET_USE_MKL_LAYERNORM=1)
endif()
if(USE_ONEDNN)
  # CPU architecture (e.g., C5) can't run on another architecture (e.g., g3).
  if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /Gy")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /EHsc /Gy")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /EHsc /Gy")
    if(NOT MXNET_FORCE_SHARED_CRT)
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
      set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
    endif()
  endif()

  function(load_onednn)
    set(MKLDNN_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
    set(MKLDNN_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
    set(MKLDNN_ARCH_OPT_FLAGS "" CACHE INTERNAL "" FORCE)
    set(MKLDNN_ENABLE_JIT_PROFILING OFF CACHE INTERNAL "" FORCE)
    set(MKLDNN_LIBRARY_TYPE STATIC CACHE INTERNAL "" FORCE)
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE INTERNAL "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE INTERNAL "" FORCE)

    if(NOT USE_OPENMP)
      set(MKLDNN_CPU_RUNTIME SEQ CACHE INTERNAL "" FORCE)
    endif()

    set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/onednn")
    add_subdirectory(3rdparty/onednn)
  endfunction()
  load_onednn()
  include_directories(3rdparty/onednn/include)
  include_directories(${PROJECT_BINARY_DIR}/3rdparty/onednn/include)
  add_definitions(-DMXNET_USE_ONEDNN=1)
  list(APPEND mxnet_LINKER_LIBS dnnl)
  set_target_properties(dnnl PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
endif()

if(USE_CPP_PACKAGE)
    add_definitions(-DMXNET_USE_CPP_PACKAGE=1)
endif()

if(USE_INTGEMM)
  message(STATUS "Using intgemm")
  add_subdirectory(3rdparty/intgemm EXCLUDE_FROM_ALL)
  add_definitions(-DMXNET_USE_INTGEMM=1)
endif()

# Allow Cuda compiles outside of src tree to find things in 'src' and 'include'
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

cmake_dependent_option(USE_INT64_TENSOR_SIZE "Use int64_t to represent the total number of elements in a tensor" ON "CMAKE_SIZEOF_VOID_P EQUAL 8" OFF)

include(cmake/ChooseBlas.cmake)

if(USE_ASAN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
  set(GTEST_LIBRARIES "${GTEST_LIBRARIES} -fsanitize=address")
  list(APPEND mxnet_LINKER_LIBS asan)
endif()

list(APPEND mxnet_LINKER_LIBS ${mshadow_LINKER_LIBS})
message("After choosing blas, linking to ${mxnet_LINKER_LIBS}")

foreach(var ${C_CXX_INCLUDE_DIRECTORIES})
    include_directories(${var})
endforeach()

include_directories("include")
include_directories("3rdparty/tvm/nnvm/include")
include_directories("3rdparty/tvm/include")
include_directories("3rdparty/dmlc-core/include")
include_directories("3rdparty/dlpack/include")

if(UNIX)
  find_library(RTLIB rt)
  if(RTLIB)
    list(APPEND mxnet_LINKER_LIBS ${RTLIB})
  endif()
endif()

set(ALT_MALLOC_FLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

# ---[ gperftools
if(USE_GPERFTOOLS)
  find_package(Gperftools)
  if(GPERFTOOLS_FOUND)
    message(STATUS "Using Gperftools malloc (tcmalloc)")
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${GPERFTOOLS_LIBRARIES})
    set(USE_JEMALLOC 0)
  endif()
endif()

# ---[ jemalloc
if(USE_JEMALLOC)
  if(GPERFTOOLS_FOUND)
    message(ERROR " Only one of USE_JEMALLOC and USE_GPERFTOOLS can be defined at once")
  endif()
  find_package(JeMalloc)
  if(JEMALLOC_FOUND)
    message(STATUS "Using JEMalloc malloc")
    add_definitions(-DUSE_JEMALLOC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALT_MALLOC_FLAGS}")
    include_directories(${JEMALLOC_INCLUDE_DIRS})
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${JEMALLOC_LIBRARIES})
  endif()
endif()

if(USE_LIBJPEG_TURBO)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(TURBOJPEG REQUIRED libturbojpeg)
  include_directories(SYSTEM ${TURBOJPEG_INCLUDE_DIRS})
  list(APPEND mxnet_LINKER_LIBS ${TURBOJPEG_LINK_LIBRARIES})
  add_definitions(-DMXNET_USE_LIBJPEG_TURBO=1)
else()
  add_definitions(-DMXNET_USE_LIBJPEG_TURBO=0)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    message(STATUS "OpenCV imgcodecs missing")
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND mxnet_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV ${OpenCV_VERSION} found (${OpenCV_CONFIG_PATH})")
  message(STATUS " OpenCV_LIBS=${OpenCV_LIBS}")
  add_definitions(-DMXNET_USE_OPENCV=1)
else(USE_OPENCV)
  message(STATUS "OpenCV Disabled")
  add_definitions(-DMXNET_USE_OPENCV=0)
endif()

# ---[ OpenMP
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # Enable pragma omp simd
    # "While the name of this switch is 'experimental', the switch itself, and
    # the functionality it enables is fully supported and production-ready.
    # The name reflects that it doesn’t enable any complete subset or
    # version of an OpenMP standard."
    # -- https://devblogs.microsoft.com/cppblog/simd-extension-to-c-openmp-in-visual-studio/
    if(MSVC)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -openmp:experimental")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:experimental")
    endif()
    if(NOT BLAS STREQUAL "MKL")
      # Linker flags for Intel OMP are already set in case MKL is used. Only set if not MKL
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
    add_definitions(-DMXNET_USE_OPENMP=1)
  endif()
elseif(UNIX)
  # Prefer compiler pthread flag. This is the recommended way, but not backwards
  # compatible and thus not the cmake default.
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  list(APPEND mxnet_LINKER_LIBS Threads::Threads)
endif()

# ---[ LAPack
if(USE_LAPACK)
  message("USE_LAPACK is ON")
  add_definitions(-DMXNET_USE_LAPACK=1)
  if(NOT USE_LAPACKE_INTERFACE)
    # BLAS=open case is handled in ChooseBlas.cmake
    if(NOT MSVC AND NOT CMAKE_BUILD_TYPE STREQUAL "Distribution"
       AND NOT BLAS STREQUAL "Open" AND NOT BLAS STREQUAL "open")
      list(APPEND mxnet_LINKER_LIBS lapack)
    endif()
  endif()
endif()

# ---[ jemalloc
if(USE_JEMALLOC)
  find_package(JeMalloc)
  if(JEMALLOC_FOUND)
    add_definitions(-DUSE_JEMALLOC)
    include_directories(${JEMALLOC_INCLUDE_DIRS})
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${JEMALLOC_LIBRARIES})
  endif()
endif()

include(CTest)
set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest/googletest")
set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_LIBRARY gtest)

add_subdirectory(${GTEST_ROOT})
set_target_properties(gtest PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
set_target_properties(gtest_main PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
find_package(GTest REQUIRED)

# cudnn detection
if(USE_CUDNN)
  find_package(CUDNN)
  if(CUDNN_FOUND)
    add_definitions(-DUSE_CUDNN)
    include_directories(SYSTEM ${CUDNN_INCLUDE})
    list(APPEND mxnet_LINKER_LIBS ${CUDNN_LIBRARY})
  else()
    set(USE_CUDNN OFF)
  endif()
endif()

# cutensor detection
if(USE_CUTENSOR)
  find_package(CUTENSOR)
  if(CUTENSOR_FOUND)
    add_definitions(-DUSE_CUTENSOR)
    include_directories(SYSTEM ${CUTENSOR_INCLUDE})
    list(APPEND mxnet_LINKER_LIBS ${CUTENSOR_LIBRARY})
  else()
    set(USE_CUTENSOR OFF)
  endif()
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/cmake)
  add_subdirectory("3rdparty/dmlc-core")
  set_target_properties(dmlc PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
endif()

FILE(GLOB_RECURSE SOURCE "src/*.cc" "src/*.h" "include/*.h")
FILE(GLOB_RECURSE CUDA "src/*.cu" "src/*.cuh")

if(MSVC)
  FILE(GLOB_RECURSE TVM_BRIDGE_SOURCE "src/*/tvm_bridge.cc")
  list(REMOVE_ITEM SOURCE ${TVM_BRIDGE_SOURCE})
endif()

if(NOT USE_INTGEMM)
  FILE(GLOB_RECURSE INTGEMM_OPERATOR_SOURCE "src/operator/contrib/intgemm/*.cc" "src/operator/contrib/intgemm/*.h")
  list(REMOVE_ITEM SOURCE ${INTGEMM_OPERATOR_SOURCE})
endif()

# add nnvm to source
FILE(GLOB_RECURSE NNVMSOURCE
  3rdparty/tvm/nnvm/src/c_api/*.cc
  3rdparty/tvm/nnvm/src/core/*.cc
  3rdparty/tvm/nnvm/src/pass/*.cc
  3rdparty/tvm/nnvm/src/c_api/*.h
  3rdparty/tvm/nnvm/src/core/*.h
  3rdparty/tvm/nnvm/src/pass/*.h
  3rdparty/tvm/nnvm/include/*.h)
add_library(nnvm OBJECT ${NNVMSOURCE})
set_target_properties(nnvm PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
list(APPEND SOURCE $<TARGET_OBJECTS:nnvm>)

add_library(miniz STATIC "3rdparty/miniz/miniz.c")
target_include_directories(miniz PUBLIC "3rdparty/miniz")
list(APPEND mxnet_LINKER_LIBS miniz)

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc" "3rdparty/tvm/nnvm/*.cc" "plugin/*.cc")
FILE(GLOB_RECURSE GROUP_Include "src/*.h" "3rdparty/tvm/nnvm/*.h" "3rdparty/mshadow/mshadow/*.h" "plugin/*.h")
FILE(GLOB_RECURSE GROUP_CUDA "src/*.cu" "src/*.cuh" "3rdparty/mshadow/mshadow/*.cuh" "plugin/*.cu"
  "plugin/*.cuh" "3rdparty/nvidia_cub/cub/*.cuh")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_Include})
assign_source_group("CUDA" ${GROUP_CUDA})

if(USE_PLUGINS_WARPCTC)
    set(WARPCTC_INCLUDE  "" CACHE PATH "WARPCTC include")
    set(WARPCTC_LIB_DEBUG  "" CACHE FILEPATH "WARPCTC lib")
    set(WARPCTC_LIB_RELEASE  "" CACHE FILEPATH "WARPCTC lib")
    include_directories(SYSTEM ${WARPCTC_INCLUDE})
    list(APPEND mxnet_LINKER_LIBS ${WARPCTC_LIB})
    FILE(GLOB_RECURSE PLUGINS_SOURCE "plugin/warpctc/*.cc" "plugin/warpctc/*.h")
    FILE(GLOB_RECURSE PLUGINS_CUSRC "plugin/warpctc/*.cu")
    list(APPEND SOURCE ${PLUGINS_SOURCE})
    list(APPEND CUDA ${PLUGINS_CUSRC})
endif()

if(USE_OPERATOR_TUNING AND USE_OPENMP)
  add_definitions(-DMXNET_USE_OPERATOR_TUNING=1)
endif()

if (NOT (EXTRA_OPERATORS STREQUAL ""))
    mxnet_source_group("Extra"   GLOB_RECURSE "${EXTRA_OPERATORS}/*.cc")
    mxnet_source_group("Extra\\Cuda"   GLOB_RECURSE "${EXTRA_OPERATORS}/*.cu")
    FILE(GLOB_RECURSE EXTRA_SRC "${EXTRA_OPERATORS}/*.cc")
    FILE(GLOB_RECURSE EXTRA_CUSRC "${EXTRA_OPERATORS}/*.cu")
    list(APPEND SOURCE ${EXTRA_SRC} ${EXTRA_CUSRC})
endif()

if(MSVC AND NOT MXNET_FORCE_SHARED_CRT)
  foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    elseif(${flag_var} MATCHES "/MDd")
      string(REGEX REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")    
    endif()
  endforeach(flag_var)
endif()

if(USE_CUDA)
  # CUDA_SELECT_NVCC_ARCH_FLAGS is not deprecated, though part of deprecated
  # FindCUDA https://gitlab.kitware.com/cmake/cmake/issues/19199
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/upstream/select_compute_arch.cmake)
  CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${MXNET_CUDA_ARCH})
  message("-- CUDA: Using the following NVCC architecture flags ${CUDA_ARCH_FLAGS}")
  set(arch_code_list)
  foreach(arch_str ${CUDA_ARCH_FLAGS})
    if((arch_str MATCHES ".*sm_[0-9]+"))
      string( REGEX REPLACE  ".*sm_([0-9]+)" "\\1" arch_code ${arch_str} )
      list(APPEND arch_code_list ${arch_code})
    endif()
  endforeach()

  string(REPLACE ";" " " CUDA_ARCH_FLAGS_SPACES "${CUDA_ARCH_FLAGS}")

  find_package(CUDAToolkit REQUIRED cublas cufft cusolver curand nvrtc cuda_driver
    OPTIONAL_COMPONENTS nvToolsExt)

  list(APPEND mxnet_LINKER_LIBS CUDA::cudart CUDA::cublas CUDA::cufft CUDA::cusolver CUDA::curand
                                CUDA::nvrtc CUDA::cuda_driver)
  list(APPEND SOURCE ${CUDA})
  add_definitions(-DMXNET_USE_CUDA=1)

  if(UNIX)
    if(USE_NVML)
      find_package(NVML)
      if(NVML_FOUND)
        include_directories(${NVML_INCLUDE_DIRS})
        list(APPEND mxnet_LINKER_LIBS ${NVML_LIBRARIES})
        add_definitions(-DMXNET_USE_NVML=1)
      else()
        add_definitions(-DMXNET_USE_NVML=0)
        message(WARNING "Could not find NVML libraries")
      endif()
    endif()
  endif()
  if(USE_NCCL)
    find_package(NCCL)
    if(NCCL_FOUND)
      include_directories(${NCCL_INCLUDE_DIRS})
      list(APPEND mxnet_LINKER_LIBS ${NCCL_LIBRARIES})
      add_definitions(-DMXNET_USE_NCCL=1)
    else()
      add_definitions(-DMXNET_USE_NCCL=0)
      message(WARNING "Could not find NCCL libraries")
    endif()
  endif()
  if(UNIX)
    if(USE_NVTX AND CUDA_nvToolsExt_LIBRARY)
      list(APPEND mxnet_LINKER_LIBS CUDA::nvToolsExt)
      add_definitions(-DMXNET_USE_NVTX=1)
    else()
      message("Building without NVTX support.")
    endif()
  endif()

  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  link_directories(${CUDAToolkit_LIBRARY_DIR})
endif()

if(CUDAToolkit_VERSION_MAJOR LESS "11")
  include_directories("3rdparty/nvidia_cub")
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /Gy")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /EHsc /Gy")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /EHsc /Gy")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")
  set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL} /OPT:REF /OPT:ICF")
  set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF /OPT:ICF")

endif()

# Add cmake targets
add_subdirectory("3rdparty/mshadow")

if(USE_CUDA AND USE_FATBIN_COMPRESSION)
  string(APPEND CMAKE_CUDA_FLAGS " --fatbin-options --compress-all")
  message("-- CUDA: Adding NVCC options: --fatbin-options --compress-all")
endif()
if(UNIX)
  string(APPEND CMAKE_CUDA_FLAGS " ${CUDA_ARCH_FLAGS_SPACES}")
  # Create dummy file since we want an empty shared library before linking
  set(DUMMY_SOURCE ${CMAKE_BINARY_DIR}/dummy.c)
  file(WRITE ${DUMMY_SOURCE} "")
  if(MXNET_BUILD_SHARED_LIBS)
    add_library(mxnet SHARED ${SOURCE})
  else()
    add_library(mxnet STATIC ${SOURCE})
  endif()
  target_link_libraries(mxnet PUBLIC mshadow)
  target_link_libraries(mxnet PUBLIC ${CMAKE_DL_LIBS})
  if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(mxnet PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
    # Ignore erroneous compiler warnings:
    # 1) variables used in '#pragma omp parallel' are considered unused
    target_compile_options(mxnet PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-variable>")
    if(USE_CUDA)
      # Note: "=" is required to avoid breaking ccache
      string(APPEND CMAKE_CUDA_FLAGS " -Werror=cross-execution-space-call")
    endif()
  endif()
  if(ENABLE_TESTCOVERAGE)
    find_program(GCOV_PATH gcov REQUIRED)
    target_compile_options(mxnet PUBLIC "--coverage")
    target_link_libraries(mxnet PUBLIC gcov)
  endif()
  if(APPLE)
    set_target_properties(mxnet PROPERTIES LINK_FLAGS "-Wl,-exported_symbols_list,${PROJECT_SOURCE_DIR}/cmake/libmxnet.sym")
  else()
    set_target_properties(mxnet PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  endif()
elseif(MSVC)
  if(USE_CUDA)
    if(USE_SPLIT_ARCH_DLL)
      add_executable(gen_warp tools/windowsbuild/gen_warp.cpp)
      add_library(mxnet SHARED tools/windowsbuild/warp_dll.cpp ${CMAKE_BINARY_DIR}/warp_gen_cpp.cpp
                  ${CMAKE_BINARY_DIR}/warp_gen.asm)
      target_link_libraries(mxnet PRIVATE cudart Shlwapi)
      list(GET arch_code_list 0 mxnet_first_arch)
      foreach(arch ${arch_code_list})
        add_library(mxnet_${arch} SHARED ${SOURCE})
        target_link_libraries(mxnet_${arch} PUBLIC mshadow)
        target_compile_definitions(mxnet_${arch} PRIVATE -DWIN32_LEAN_AND_MEAN)
        target_compile_options(
          mxnet_${arch}
          PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_${arch}>"
        )
        target_compile_options(
          mxnet_${arch}
          PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_${arch},compute_${arch}>"
        )
        if(MXNET_FORCE_SHARED_CRT)
          target_compile_options(
            mxnet_${arch} 
            PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MDd -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")          
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:MINSIZEREL>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
        else()
          target_compile_options(
            mxnet_${arch} 
            PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MTd -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")          
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:MINSIZEREL>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
        endif()
      endforeach()

      add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/warp_gen_cpp.cpp ${CMAKE_BINARY_DIR}/warp_gen.asm
        COMMAND gen_warp $<TARGET_FILE:mxnet_${mxnet_first_arch}> WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ DEPENDS $<TARGET_FILE:mxnet_${mxnet_first_arch}>)
    else(USE_SPLIT_ARCH_DLL)
      string(REPLACE ";" " " NVCC_FLAGS_ARCH "${NVCC_FLAGS_ARCH}")
      string(APPEND CMAKE_CUDA_FLAGS " ${CUDA_ARCH_FLAGS_SPACES}")
      add_library(mxnet SHARED ${SOURCE})
      target_compile_definitions(mxnet PRIVATE -DWIN32_LEAN_AND_MEAN)
      target_link_libraries(mxnet PUBLIC mshadow)
      if(MXNET_FORCE_SHARED_CRT)
        target_compile_options(
            mxnet 
            PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MDd -Gy /bigobj>")
        target_compile_options(
            mxnet
            PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
        target_compile_options(
            mxnet
            PRIVATE "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
        target_compile_options(
            mxnet
            PRIVATE "$<$<AND:$<CONFIG:MINSIZEREL>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
      else()
        target_compile_options(
          mxnet_${arch} 
          PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MTd -Gy /bigobj>")
        target_compile_options(
          mxnet_${arch}
          PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")          
        target_compile_options(
          mxnet_${arch}
          PRIVATE "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
        target_compile_options(
          mxnet_${arch}
          PRIVATE "$<$<AND:$<CONFIG:MINSIZEREL>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
      endif()
    endif(USE_SPLIT_ARCH_DLL)
  else()
    add_library(mxnet SHARED ${SOURCE})
    target_compile_definitions(mxnet PRIVATE -DWIN32_LEAN_AND_MEAN)
    target_link_libraries(mxnet PUBLIC mshadow)
  endif()
endif()
target_compile_definitions(mxnet PUBLIC DMLC_LOG_FATAL_THROW=$<BOOL:${LOG_FATAL_THROW}>)

# extension libraries (custom operators, custom subgraphs) are built by default
add_library(customop_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/gemm_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
add_library(transposecsr_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/transposecsr_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
add_library(transposerowsp_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/transposerowsp_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
add_library(subgraph_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_subgraph/subgraph_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
add_library(pass_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_pass/pass_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)

if(IS_DIRECTORY ${BUILD_EXTENSION_PATH})
  if(MSVC)
    message(FATAL_ERROR "Windows builds are not support for external ops")
  else()
    add_subdirectory(${BUILD_EXTENSION_PATH} ${BUILD_EXTENSION_PATH}/build)
  endif()
endif()

target_include_directories(customop_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(transposecsr_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(transposerowsp_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(subgraph_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(pass_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
if(USE_CUDA)
  add_library(customop_gpu_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/relu_lib.cu ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/relu_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
  target_include_directories(customop_gpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op)
endif()
if(UNIX)
  if (USE_CUDA)
    target_compile_options(customop_gpu_lib PUBLIC -shared)
  endif()
elseif(MSVC)
  target_compile_options(customop_lib PUBLIC /LD)
  target_compile_options(transposecsr_lib PUBLIC /LD)
  target_compile_options(transposerowsp_lib PUBLIC /LD)
  target_compile_options(subgraph_lib PUBLIC /LD)
  target_compile_options(pass_lib PUBLIC /LD)
  set_target_properties(customop_lib PROPERTIES PREFIX "lib")
  set_target_properties(transposecsr_lib PROPERTIES PREFIX "lib")
  set_target_properties(transposerowsp_lib PROPERTIES PREFIX "lib")
  set_target_properties(subgraph_lib PROPERTIES PREFIX "lib")
  set_target_properties(pass_lib PROPERTIES PREFIX "lib")
  if(USE_CUDA)
    if(MXNET_FORCE_SHARED_CRT)
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-LD -MD>")
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/LD>")
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/MD>")
    else()
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-LD -MT>")
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/LD>")
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/MT>")
    endif()
    set_target_properties(customop_gpu_lib PROPERTIES PREFIX "lib")
  endif()
endif()

if(USE_DIST_KVSTORE)
  add_subdirectory("3rdparty/ps-lite")
  add_definitions(-DMXNET_USE_DIST_KVSTORE)
  list(APPEND mxnet_LINKER_LIBS pslite)
  set_target_properties(pslite PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
endif()

if(USE_ONEDNN)
    add_custom_command(TARGET mxnet POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/3rdparty/onednn/include/oneapi/dnnl/dnnl_config.h  ${CMAKE_SOURCE_DIR}/include/onednn/oneapi/dnnl/
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/3rdparty/onednn/include/oneapi/dnnl/dnnl_version.h  ${CMAKE_SOURCE_DIR}/include/onednn/oneapi/dnnl/)
endif()

if(USE_INTGEMM)
  target_link_libraries(mxnet PRIVATE intgemm)
endif()

function(BuildTVMOP)
  # scope the variables in BuildTVM.cmake to avoid conflict
  include(cmake/BuildTVM.cmake)
  add_subdirectory("3rdparty/tvm")
  set_target_properties(tvm PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
  set_target_properties(tvm_runtime PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
endfunction()

if(USE_TVM_OP)
  list(APPEND mxnet_LINKER_LIBS tvm_runtime)
  BuildTVMOP()
  find_package(Python3 REQUIRED)
  set(TVM_OP_COMPILE_OPTIONS "-o${CMAKE_CURRENT_BINARY_DIR}" "--config" "${CMAKE_CURRENT_BINARY_DIR}/tvmop.conf" "-L" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm")
  if(USE_CUDA)
    set(TVM_OP_COMPILE_OPTIONS "${TVM_OP_COMPILE_OPTIONS}" "--cuda-arch" "\"${CUDA_ARCH_FLAGS}\"")
  endif()

  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH="${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/python:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/topi/python:${CMAKE_CURRENT_SOURCE_DIR}/contrib"
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm:$ENV{LD_LIBRARY_PATH}
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/contrib/tvmop/compile.py ${TVM_OP_COMPILE_OPTIONS}
  )
endif()

if(USE_PLUGINS_WARPCTC)
  list(APPEND mxnet_LINKER_LIBS ${WARPCTC_LIB})
endif()

if(MSVC)
  if(USE_SPLIT_ARCH_DLL AND USE_CUDA)
    foreach(arch ${arch_code_list})
      target_link_libraries(mxnet_${arch} PUBLIC ${mxnet_LINKER_LIBS})
      target_link_libraries(mxnet_${arch} PUBLIC dmlc)
    endforeach()
  endif()
endif()

target_link_libraries(mxnet PUBLIC ${mxnet_LINKER_LIBS})
target_link_libraries(mxnet PUBLIC dmlc)

if(USE_OPENCV AND OpenCV_VERSION_MAJOR GREATER 2)
  add_executable(im2rec "tools/im2rec.cc")
  target_link_libraries(im2rec
    ${mxnet_LINKER_LIBS}
    ${OpenCV_LIBS}
    mxnet
    dmlc
    )
else()
    message(WARNING "OpenCV_VERSION_MAJOR: ${OpenCV_VERSION_MAJOR}, version 3 with imgcodecs \
    is required for im2rec, im2rec will not be available")
endif()


if(MSVC AND USE_MXNET_LIB_NAMING)
  set_target_properties(mxnet PROPERTIES OUTPUT_NAME "libmxnet")
endif()


# NOTE: Public headers will be installed into ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}, see
#       https://cmake.org/cmake/help/v3.13/variable/CMAKE_INSTALL_PREFIX.html
#       https://cmake.org/cmake/help/v3.13/module/GNUInstallDirs.html
include(GNUInstallDirs)
install(TARGETS mxnet
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT   MXNET_Runtime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT            MXNET_Runtime
          NAMELINK_COMPONENT   MXNET_Development
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT   MXNET_Development
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mshadow/mshadow/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mshadow)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mxnet)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/nnvm/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if (INSTALL_EXAMPLES)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/example  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
endif()

if (USE_SIGNAL_HANDLER)
    add_definitions(-DMXNET_USE_SIGNAL_HANDLER=1)
endif()

# AUTO_INSTALL_DIR -> Optional: specify post-build install direcory
if(AUTO_INSTALL_DIR)
  # ---[ Install Includes
  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${AUTO_INSTALL_DIR}/include
    )

  # ---[ Install Examples
  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/example ${AUTO_INSTALL_DIR}/example
    )
endif()

if(INSTALL_PYTHON_VERSIONS)
  message(STATUS "Installing for python versions: ${INSTALL_PYTHON_VERSIONS}")
  foreach(version ${INSTALL_PYTHON_VERSIONS})
    set(outdir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python${version}/site-packages/mxnet)
    add_custom_command(TARGET mxnet POST_BUILD
      COMMAND mkdir -p ${outdir}
      COMMAND cp -ru ${CMAKE_CURRENT_SOURCE_DIR}/python/mxnet/* ${outdir}
      )
  endforeach()
endif()

if(USE_CPP_PACKAGE)
  add_subdirectory(cpp-package)
  target_compile_definitions(mxnet PUBLIC MXNET_USE_CPP_PACKAGE=1)
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Distribution")
  # Staticbuild applies linker version script to hide private symbols, breaking unit tests
  add_subdirectory(tests)
endif()

# ---[ Linter target
find_package(Python3)
set(LINT_DIRS "include src plugin tests")
set(EXCLUDE_PATH "src/operator/contrib/ctc_include")
add_custom_target(mxnet_lint COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE} -DLINT_DIRS=${LINT_DIRS} -DPROJECT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DPROJECT_NAME=mxnet -DEXCLUDE_PATH=${EXCLUDE_PATH} -P ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/cmake/lint.cmake)

if(BUILD_CYTHON_MODULES)
  include(cmake/BuildCythonModules.cmake)
  add_cython_modules(3) # Build cython module for python3 if python3 is found
  if(NOT ${PYTHON3_FOUND})
    message(FATAL_ERROR "No python interpreter found to build cython modules")
  endif()
endif()

# https://github.com/apache/incubator-mxnet/issues/20145
if(UNIX)
    if(USE_NVML)
      find_package(NVML)
      if(NVML_FOUND)
         target_compile_definitions(mxnet PRIVATE NVML_NO_UNVERSIONED_FUNC_DEFS)
    endif()
  endif()
endif()
