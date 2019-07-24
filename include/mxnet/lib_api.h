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

/*!
 * Copyright (c) 2015 by Contributors
 * \file lib_api.h
 * \brief APIs to interact with libraries
 */
#ifndef MXNET_LIB_API_H_
#define MXNET_LIB_API_H_

#include <cstdlib>

#define INITIALIZE_STR "initialize"
typedef int (*initialize_t)(int);

extern "C" {
    /*
    Function: initialize
    Parameters:
    - int : MXNet version passed to library
    Returns:
    - int : Non-zero value on error i.e. library incompatible with passed MXNet version
    */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
    __declspec(dllexport) int __cdecl initialize(int);
#else
    int initialize(int);
#endif
}
#endif  // MXNET_LIB_API_H_
