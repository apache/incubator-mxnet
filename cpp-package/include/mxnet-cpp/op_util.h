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
*  Copyright (c) 2017 by Contributors
* \file op_util.h
* \brief operator helper functions
* \author Chris Olivier
*/

#ifndef MXNET_CPP_OP_UTIL_H_
#define MXNET_CPP_OP_UTIL_H_

#include <string>

#if defined(MXNET_USE_CAFFE) && MXNET_USE_CAFFE != 0
#include <caffe/proto/caffe.pb.h>
#include <google/protobuf/text_format.h>
#endif

namespace mxnet {
namespace cpp {

#if defined(MXNET_USE_CAFFE) && MXNET_USE_CAFFE != 0

inline ::caffe::LayerParameter textToCaffeLayerParameter(const std::string& text) {
  caffe::NetParameter np;
  const bool success = google::protobuf::TextFormat::ParseFromString(text, &np);
  CHECK_EQ(success, true) << "Invalid protpbuf layer string: " << text;
  return ::caffe::LayerParameter(np.layer(0));
}

template<typename StreamType>
inline StreamType& operator << (StreamType& os, const ::caffe::LayerParameter& op) {
  std::string s;
  caffe::NetParameter np;
  // Avoid wasting time making a copy -- just push in out default object's pointer
  np.mutable_layer()->AddAllocated(const_cast<::caffe::LayerParameter *>(&op));
  google::protobuf::TextFormat::PrintToString(np, &s);
  np.mutable_layer()->ReleaseLast();
  os << s;
  return os;
}
#endif

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_OP_UTIL_H_
