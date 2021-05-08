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
*  Copyright (c) 2016 by Contributors
* \file base.h
* \brief base definitions for mxnetcpp
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_CPP_BASE_H_
#define MXNET_CPP_BASE_H_

#include <cstdlib>
#include "mxnet/c_api.h"
#include "nnvm/c_api.h"

namespace mxnet {
namespace cpp {

typedef unsigned index_t;

enum OpReqType {
  /*! \brief no operation, do not write anything */
  kNullOp,
  /*! \brief write gradient to provided space */
  kWriteTo,
  /*!
  * \brief perform an inplace write,
  * Target shares memory with one of input arguments.
  * This option only happen when
  */
  kWriteInplace,
  /*! \brief add to the provided space */
  kAddTo
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_BASE_H_
