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
 * \file ndarray.hpp
 * \brief implementation of the ndarray
 * \author Zhang Chen, Chuntao Hong
 */

#ifndef MXNET_CPP_NDARRAY_HPP_
#define MXNET_CPP_NDARRAY_HPP_

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <iterator>
#include "dmlc/logging.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/operator.h"

namespace mxnet {
namespace cpp {

inline NDArray::NDArray() {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
inline NDArray::NDArray(const NDArrayHandle &handle) {
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
inline NDArray::NDArray(const std::vector<mx_uint> &shape, const Context &context,
                        bool delay_alloc, int dtype) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateEx(shape.data(), shape.size(), context.GetDeviceType(),
                             context.GetDeviceId(), delay_alloc, dtype, &handle),
           0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
inline NDArray::NDArray(const Shape &shape, const Context &context,
                        bool delay_alloc, int dtype) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateEx(shape.data(), shape.ndim(), context.GetDeviceType(),
                             context.GetDeviceId(), delay_alloc, dtype, &handle),
           0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
inline NDArray::NDArray(const mx_float *data, size_t size) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  MXNDArraySyncCopyFromCPU(handle, data, size);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
inline NDArray::NDArray(const mx_float *data, const Shape &shape,
                        const Context &context) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreate(shape.data(), shape.ndim(), context.GetDeviceType(),
                           context.GetDeviceId(), false, 0, &handle),
           0);
  CHECK_EQ(MXNDArraySyncCopyFromCPU(handle, data, shape.Size()), 0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
inline NDArray::NDArray(const std::vector<mx_float> &data, const Shape &shape,
                        const Context &context) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreate(shape.data(), shape.ndim(), context.GetDeviceType(),
                           context.GetDeviceId(), false, 0, &handle),
           0);
  MXNDArraySyncCopyFromCPU(handle, data.data(), shape.Size());
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
inline NDArray::NDArray(const std::vector<mx_float> &data) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  MXNDArraySyncCopyFromCPU(handle, data.data(), data.size());
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}

inline NDArray NDArray::operator+(mx_float scalar) {
  NDArray ret;
  Operator("_plus_scalar")(*this, scalar).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator-(mx_float scalar) {
  NDArray ret;
  Operator("_minus_scalar")(*this, scalar).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator*(mx_float scalar) {
  NDArray ret;
  Operator("_mul_scalar")(*this, scalar).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator/(mx_float scalar) {
  NDArray ret;
  Operator("_div_scalar")(*this, scalar).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator%(mx_float scalar) {
  NDArray ret;
  Operator("_mod_scalar")(*this, scalar).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator+(const NDArray &rhs) {
  NDArray ret;
  Operator("_plus")(*this, rhs).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator-(const NDArray &rhs) {
  NDArray ret;
  Operator("_minus")(*this, rhs).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator*(const NDArray &rhs) {
  NDArray ret;
  Operator("_mul")(*this, rhs).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator/(const NDArray &rhs) {
  NDArray ret;
  Operator("_div")(*this, rhs).Invoke(ret);
  return ret;
}
inline NDArray NDArray::operator%(const NDArray &rhs) {
  NDArray ret;
  Operator("_mod")(*this, rhs).Invoke(ret);
  return ret;
}
inline NDArray &NDArray::operator=(mx_float scalar) {
  Operator("_set_value")(scalar).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator+=(mx_float scalar) {
  Operator("_plus_scalar")(*this, scalar).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator-=(mx_float scalar) {
  Operator("_minus_scalar")(*this, scalar).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator*=(mx_float scalar) {
  Operator("_mul_scalar")(*this, scalar).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator/=(mx_float scalar) {
  Operator("_div_scalar")(*this, scalar).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator%=(mx_float scalar) {
  Operator("_mod_scalar")(*this, scalar).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator+=(const NDArray &rhs) {
  Operator("_plus")(*this, rhs).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator-=(const NDArray &rhs) {
  Operator("_minus")(*this, rhs).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator*=(const NDArray &rhs) {
  Operator("_mul")(*this, rhs).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator/=(const NDArray &rhs) {
  Operator("_div")(*this, rhs).Invoke(*this);
  return *this;
}
inline NDArray &NDArray::operator%=(const NDArray &rhs) {
  Operator("_mod")(*this, rhs).Invoke(*this);
  return *this;
}

inline NDArray NDArray::ArgmaxChannel() {
  NDArray ret;
  Operator("argmax_channel")(*this).Invoke(ret);
  return ret;
}

inline void NDArray::SyncCopyFromCPU(const mx_float *data, size_t size) {
  MXNDArraySyncCopyFromCPU(blob_ptr_->handle_, data, size);
}
inline void NDArray::SyncCopyFromCPU(const std::vector<mx_float> &data) {
  MXNDArraySyncCopyFromCPU(blob_ptr_->handle_, data.data(), data.size());
}
inline void NDArray::SyncCopyToCPU(mx_float *data, size_t size) {
  MXNDArraySyncCopyToCPU(blob_ptr_->handle_, data, size > 0 ? size : Size());
}
inline void NDArray::SyncCopyToCPU(std::vector<mx_float> *data, size_t size) {
  size = size > 0 ? size : Size();
  data->resize(size);
  MXNDArraySyncCopyToCPU(blob_ptr_->handle_, data->data(), size);
}
inline NDArray NDArray::Copy(const Context &ctx) const {
  NDArray ret(GetShape(), ctx, true, this->GetDType());
  Operator("_copyto")(*this).Invoke(ret);
  return ret;
}
inline NDArray NDArray::CopyTo(NDArray * other) const {
  Operator("_copyto")(*this).Invoke(*other);
  return *other;
}
inline NDArray NDArray::Slice(mx_uint begin, mx_uint end) const {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArraySlice(GetHandle(), begin, end, &handle), 0);
  return NDArray(handle);
}
inline NDArray NDArray::Reshape(const Shape &new_shape) const {
  NDArrayHandle handle;
  std::vector<int> dims(new_shape.ndim());
  for (index_t i = 0; i < new_shape.ndim(); ++i) {
    dims[i] = new_shape[i];
  }
  new_shape.data();
  CHECK_EQ(
      MXNDArrayReshape(GetHandle(), new_shape.ndim(), dims.data(), &handle), 0);
  return NDArray(handle);
}
inline void NDArray::WaitToRead() const {
  CHECK_EQ(MXNDArrayWaitToRead(blob_ptr_->handle_), 0) << MXGetLastError();
}
inline void NDArray::WaitToWrite() {
  CHECK_EQ(MXNDArrayWaitToWrite(blob_ptr_->handle_), 0) << MXGetLastError();
}
inline void NDArray::WaitAll() { CHECK_EQ(MXNDArrayWaitAll(), 0) << MXGetLastError(); }
inline void NDArray::SampleGaussian(mx_float mu, mx_float sigma, NDArray *out) {
  Operator("_random_normal")(mu, sigma).Invoke(*out);
}
inline void NDArray::SampleUniform(mx_float begin, mx_float end, NDArray *out) {
  Operator("_random_uniform")(begin, end).Invoke(*out);
}
inline void NDArray::Load(const std::string &file_name,
                          std::vector<NDArray> *array_list,
                          std::map<std::string, NDArray> *array_map) {
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoad(file_name.c_str(), &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  if (array_list != nullptr) {
    array_list->reserve(out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      array_list->push_back(NDArray(out_arr[i]));
    }
  }
  if (array_map != nullptr && out_name_size > 0) {
    CHECK_EQ(out_name_size, out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      (*array_map)[out_names[i]] = NDArray(out_arr[i]);
    }
  }
}
inline std::map<std::string, NDArray> NDArray::LoadToMap(
    const std::string &file_name) {
  std::map<std::string, NDArray> array_map;
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoad(file_name.c_str(), &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  if (out_name_size > 0) {
    CHECK_EQ(out_name_size, out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      array_map[out_names[i]] = NDArray(out_arr[i]);
    }
  }
  return array_map;
}
inline std::vector<NDArray> NDArray::LoadToList(const std::string &file_name) {
  std::vector<NDArray> array_list;
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoad(file_name.c_str(), &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  array_list.reserve(out_size);
  for (mx_uint i = 0; i < out_size; ++i) {
    array_list.push_back(NDArray(out_arr[i]));
  }
  return array_list;
}
inline void NDArray::LoadFromBuffer(const void *buffer, size_t size,
                          std::vector<NDArray> *array_list,
                          std::map<std::string, NDArray> *array_map) {
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoadFromBuffer(buffer, size, &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  if (array_list != nullptr) {
    array_list->reserve(out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      array_list->push_back(NDArray(out_arr[i]));
    }
  }
  if (array_map != nullptr && out_name_size > 0) {
    CHECK_EQ(out_name_size, out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      (*array_map)[out_names[i]] = NDArray(out_arr[i]);
    }
  }
}
inline std::map<std::string, NDArray> NDArray::LoadFromBufferToMap(
    const void *buffer, size_t size) {
  std::map<std::string, NDArray> array_map;
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoadFromBuffer(buffer, size, &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  if (out_name_size > 0) {
    CHECK_EQ(out_name_size, out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      array_map[out_names[i]] = NDArray(out_arr[i]);
    }
  }
  return array_map;
}
inline std::vector<NDArray> NDArray::LoadFromBufferToList(const void *buffer, size_t size) {
  std::vector<NDArray> array_list;
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoadFromBuffer(buffer, size, &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  array_list.reserve(out_size);
  for (mx_uint i = 0; i < out_size; ++i) {
    array_list.push_back(NDArray(out_arr[i]));
  }
  return array_list;
}
inline void NDArray::Save(const std::string &file_name,
                          const std::map<std::string, NDArray> &array_map) {
  std::vector<NDArrayHandle> args;
  std::vector<const char *> keys;
  for (const auto &t : array_map) {
    args.push_back(t.second.GetHandle());
    keys.push_back(t.first.c_str());
  }
  CHECK_EQ(
      MXNDArraySave(file_name.c_str(), args.size(), args.data(), keys.data()),
      0);
}
inline void NDArray::Save(const std::string &file_name,
                          const std::vector<NDArray> &array_list) {
  std::vector<NDArrayHandle> args;
  for (const auto &t : array_list) {
    args.push_back(t.GetHandle());
  }
  CHECK_EQ(MXNDArraySave(file_name.c_str(), args.size(), args.data(), nullptr),
           0);
}

inline size_t NDArray::Offset(size_t h, size_t w) const {
  auto const shape = GetShape();
  CHECK_EQ(shape.size(), 2) << "The NDArray needs to be 2 dimensional.";

  return (h * shape[1]) + w;
}

inline size_t NDArray::Offset(size_t c, size_t h, size_t w) const {
  auto const shape = GetShape();
  CHECK_EQ(shape.size(), 3) << "The NDArray needs to be 3 dimensional.";
  return h * shape[0] * shape[2] + w * shape[0] + c;
}

inline mx_float NDArray::At(size_t h, size_t w) const {
  return GetData()[Offset(h, w)];
}

inline mx_float NDArray::At(size_t c, size_t h, size_t w) const {
  return GetData()[Offset(c, h, w)];
}

inline mx_float NDArray::At(size_t index) const {
  auto shape = GetShape();
  CHECK_EQ(shape.size(), 1) << "The NDArray needs to be 1 dimensional.";
  CHECK_LT(index, shape[0]) << "Specified index is out of range.";
  return GetData()[index];
}

inline size_t NDArray::Size() const {
  size_t ret = 1;
  for (auto &i : GetShape()) ret *= i;
  return ret;
}

inline std::vector<mx_uint> NDArray::GetShape() const {
  const int *out_pdata;
  int out_dim;
  MXNDArrayGetShape(blob_ptr_->handle_, &out_dim, &out_pdata);
  std::vector<mx_uint> ret;
  for (int i = 0; i < out_dim; ++i) {
    ret.push_back(out_pdata[i]);
  }
  return ret;
}

inline int NDArray::GetDType() const {
  int ret;
  MXNDArrayGetDType(blob_ptr_->handle_, &ret);
  return ret;
}

inline const mx_float *NDArray::GetData() const {
  void *ret;
  MXNDArrayGetData(blob_ptr_->handle_, &ret);
  if (GetDType() != 0) {
    return nullptr;
  }
  return static_cast<mx_float*>(ret);
}

inline Context NDArray::GetContext() const {
  int out_dev_type;
  int out_dev_id;
  MXNDArrayGetContext(blob_ptr_->handle_, &out_dev_type, &out_dev_id);
  return Context((DeviceType)out_dev_type, out_dev_id);
}

inline std::ostream & operator<<(std::ostream &out, const NDArray &ndarray) {
  // TODO(lx75249): Consider DType / beautify like numpy
  auto shape = ndarray.GetShape();
  NDArray cpu_array(ndarray.GetShape(), Context::cpu());
  if (ndarray.GetContext().GetDeviceType() != DeviceType::kGPU) {
    cpu_array = ndarray;
  } else {
    ndarray.WaitToRead();
    ndarray.CopyTo(&cpu_array);
  }

  out << '[';
  cpu_array.WaitToRead();
  std::copy(cpu_array.GetData(), cpu_array.GetData() + ndarray.Size(),
      std::ostream_iterator<float>(out, ", "));
  out << ']';
  return out;
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_NDARRAY_HPP_
