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

# Find the MKL libraries
#
# Options:
#
#   MKL_USE_SINGLE_DYNAMIC_LIBRARY  : use single dynamic library interface
#   MKL_USE_STATIC_LIBS             : use static libraries
#   MKL_MULTI_THREADED              : use multi-threading
#   MKL_USE_ILP64                   : use ilp64 data model
#   MKL_USE_CLUSTER                 : use cluster libraries
#
# This module defines the following variables:
#
#   MKL_FOUND            : True mkl is found
#   MKL_INCLUDE_DIR      : include directory
#   MKL_LIBRARIES        : the libraries to link against
#

if(MKL_FOUND)
  return()
endif()

# ---[ Options
mxnet_option(MKL_USE_SINGLE_DYNAMIC_LIBRARY "Use single dynamic library interface" ON)
mxnet_option(MKL_USE_STATIC_LIBS "Use static libraries" OFF IF NOT ${MKL_USE_SINGLE_DYNAMIC_LIBRARY})
mxnet_option(MKL_MULTI_THREADED "Use multi-threading" ON IF NOT ${MKL_USE_SINGLE_DYNAMIC_LIBRARY})
mxnet_option(MKL_USE_ILP64 "Use ilp64 data model" OFF)
mxnet_option(MKL_USE_CLUSTER "Use cluster functions" OFF IF ${CMAKE_SIZEOF_VOID_P} EQUAL 4)

# ---[ Root folders
set(INTEL_ROOT "/opt/intel" CACHE PATH "Folder contains intel libs")

if(NOT MKLROOT)
  get_filename_component(MKLROOT "${MKL_INCLUDE_DIR}" DIRECTORY)
endif()

set(MKL_INCLUDE_SEARCH_PATHS
    ${MKL_INCLUDE_SEARCH_PATHS}

    "$ENV{MKLROOT}"
    "${MKLROOT}"

    "${INTEL_ROOT}/mkl"

    /usr
    /usr/local
    )

# ---[ Find include dir
find_path(MKL_INCLUDE_DIR mkl.h
          PATHS ${MKL_INCLUDE_SEARCH_PATHS}
          PATH_SUFFIXES include)

set(MKL_LIB_SEARCH_PATHS
    ${MKL_LIB_SEARCH_PATHS}

    "$ENV{MKLROOT}"
    "${MKLROOT}"

    "${INTEL_RTL_ROOT}"
    "${INTEL_ROOT}/compiler"
    "${MKLROOT}/.."
    "${MKLROOT}/../compiler"

    /usr
    /usr/local
    )

set(__looked_for MKL_INCLUDE_DIR)

# ---[ Find libraries
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(__path_suffixes lib lib/ia32)
else()
  set(__path_suffixes lib lib64 lib/intel64)
endif()

set(__mkl_libs "")
if(MKL_USE_SINGLE_DYNAMIC_LIBRARY)
  list(APPEND __mkl_libs rt)
else()
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    if(WIN32)
      list(APPEND __mkl_libs intel_c)
    else()
      list(APPEND __mkl_libs intel)
      if(CMAKE_COMPILER_IS_GNUFORTRAN)
        list(APPEND __mkl_libs gf)
      endif()
    endif()
  else()
    set(__mkl_lib64_suffix "lp64")
    if(MKL_USE_ILP64)
      set(__mkl_lib64_suffix "ilp64")
      add_definitions(-DMKL_ILP64)
    endif()
    list(APPEND __mkl_libs "intel_${__mkl_lib64_suffix}")
    if(CMAKE_COMPILER_IS_GNUFORTRAN)
      list(APPEND __mkl_libs "gf_${__mkl_lib64_suffix}")
    endif()
  endif()

  if(MKL_MULTI_THREADED)
    list(APPEND __mkl_libs intel_thread)
  else()
    list(APPEND __mkl_libs sequential)
  endif()

  list(APPEND __mkl_libs core)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND MKL_USE_CLUSTER)
    list(APPEND __mkl_libs cdft_core)
  endif()
endif()


foreach(__lib ${__mkl_libs})
  set(__mkl_lib "mkl_${__lib}")
  string(TOUPPER ${__mkl_lib} __mkl_lib_upper)

  if(MKL_USE_STATIC_LIBS)
    set(__mkl_lib "lib${__mkl_lib}.a")
  endif()

  find_library(${__mkl_lib_upper}_LIBRARY
               NAMES ${__mkl_lib}
               PATHS ${MKL_LIB_SEARCH_PATHS}
               PATH_SUFFIXES ${__path_suffixes}
               DOC "The path to Intel(R) MKL ${__mkl_lib} library")
  mark_as_advanced(${__mkl_lib_upper}_LIBRARY)

  list(APPEND __looked_for ${__mkl_lib_upper}_LIBRARY)
  list(APPEND MKL_LIBRARIES ${${__mkl_lib_upper}_LIBRARY})
endforeach()


if(NOT MKL_USE_SINGLE_DYNAMIC_LIBRARY)
  if(MKL_USE_STATIC_LIBS)
    set(__iomp5_libs iomp5 libiomp5mt.lib)
  else()
    set(__iomp5_libs iomp5 libiomp5md.lib)
  endif()

  if(WIN32)
    find_path(INTEL_INCLUDE_DIR omp.h
              PATHS "${INTEL_ROOT}"
              PATH_SUFFIXES include)
    list(APPEND __looked_for INTEL_INCLUDE_DIR)
  endif()

  find_library(MKL_RTL_LIBRARY ${__iomp5_libs}
               PATHS ${MKL_LIB_SEARCH_PATHS}
               PATH_SUFFIXES ${__path_suffixes}
               DOC "Path to OpenMP runtime library")

  list(APPEND __looked_for MKL_RTL_LIBRARY)
  list(APPEND MKL_LIBRARIES ${MKL_RTL_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG ${__looked_for})

if(MKL_FOUND)
  message(STATUS "Found MKL (include: ${MKL_INCLUDE_DIR}, lib: ${MKL_LIBRARIES}")
endif()

mxnet_clear_vars(__looked_for __mkl_libs __path_suffixes __lib_suffix __iomp5_libs)

