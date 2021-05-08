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

cmake_minimum_required(VERSION 3.13)
project(mxnet_cpp C CXX)

add_library(mxnet_cpp INTERFACE)

set(CPP_PACKAGE_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/)
target_include_directories(mxnet_cpp INTERFACE "${CPP_PACKAGE_INCLUDE_DIR}")
file(GLOB_RECURSE CPP_PACKAGE_HEADERS
  "${CPP_PACKAGE_INCLUDE_DIR}/*.h"
  "${CPP_PACKAGE_INCLUDE_DIR}/*.hpp")
set(CPP_PACKAGE_OP_H_HEADER ${CMAKE_CURRENT_LIST_DIR}/include/mxnet-cpp/op.h)
target_sources(mxnet_cpp INTERFACE ${CPP_PACKAGE_HEADERS} ${CPP_PACKAGE_OP_H_HEADER})
target_link_libraries(mxnet_cpp INTERFACE mxnet ${mxnet_LINKER_LIBS})

add_custom_target(
  cpp_package_op_h ALL
  BYPRODUCTS ${CPP_PACKAGE_OP_H_HEADER}
  MAIN_DEPENDENCY mxnet
  DEPENDS mxnet ${CMAKE_CURRENT_SOURCE_DIR}/scripts/OpWrapperGenerator.py
  COMMAND echo "Running: OpWrapperGenerator.py"
  COMMAND python3 OpWrapperGenerator.py $<TARGET_FILE:mxnet>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
)
add_dependencies(mxnet_cpp cpp_package_op_h)

if(MSVC)
  target_compile_options(mxnet_cpp INTERFACE "/utf-8")
endif(MSVC)

if(BUILD_CPP_EXAMPLES)
  add_subdirectory(example)
  add_subdirectory(example/inference)
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
