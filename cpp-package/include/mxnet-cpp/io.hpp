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
* \file operator.hpp
* \brief implementation of data iter
* \author Zhang Chen
*/
#ifndef MXNET_CPP_IO_HPP_
#define MXNET_CPP_IO_HPP_

#include <string>
#include <vector>
#include "mxnet-cpp/io.h"

namespace mxnet {
namespace cpp {

inline MXDataIterMap*& MXDataIter::mxdataiter_map() {
    static MXDataIterMap* mxdataiter_map_ = new MXDataIterMap;
    return mxdataiter_map_;
}

inline MXDataIter::MXDataIter(const std::string &mxdataiter_type) {
  creator_ = mxdataiter_map()->GetMXDataIterCreator(mxdataiter_type);
  blob_ptr_ = std::make_shared<MXDataIterBlob>(nullptr);
}

inline void MXDataIter::BeforeFirst() {
  int r = MXDataIterBeforeFirst(blob_ptr_->handle_);
  CHECK_EQ(r, 0);
}

inline bool MXDataIter::Next() {
  int out;
  int r = MXDataIterNext(blob_ptr_->handle_, &out);
  CHECK_EQ(r, 0);
  return out;
}

inline NDArray MXDataIter::GetData() {
  NDArrayHandle handle;
  int r = MXDataIterGetData(blob_ptr_->handle_, &handle);
  CHECK_EQ(r, 0);
  return NDArray(handle);
}

inline NDArray MXDataIter::GetLabel() {
  NDArrayHandle handle;
  int r = MXDataIterGetLabel(blob_ptr_->handle_, &handle);
  CHECK_EQ(r, 0);
  return NDArray(handle);
}

inline int MXDataIter::GetPadNum() {
  int out;
  int r = MXDataIterGetPadNum(blob_ptr_->handle_, &out);
  CHECK_EQ(r, 0);
  return out;
}
inline std::vector<int> MXDataIter::GetIndex() {
  uint64_t *out_index, out_size;
  int r = MXDataIterGetIndex(blob_ptr_->handle_, &out_index, &out_size);
  CHECK_EQ(r, 0);
  std::vector<int> ret;
  for (uint64_t i = 0; i < out_size; ++i) {
    ret.push_back(out_index[i]);
  }
  return ret;
}

inline MXDataIter MXDataIter::CreateDataIter() {
  std::vector<const char *> param_keys;
  std::vector<const char *> param_values;

  for (auto &data : params_) {
    param_keys.push_back(data.first.c_str());
    param_values.push_back(data.second.c_str());
  }

  MXDataIterCreateIter(creator_, param_keys.size(), param_keys.data(),
                       param_values.data(), &blob_ptr_->handle_);
  return *this;
}

// MXDataIter MNIst

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_IO_HPP_

