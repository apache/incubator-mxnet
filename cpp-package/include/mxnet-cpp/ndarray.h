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
* \file ndarray.h
* \brief definition of ndarray
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_CPP_NDARRAY_H_
#define MXNET_CPP_NDARRAY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"

namespace mxnet {
namespace cpp {

enum DeviceType {
  kCPU = 1,
  kGPU = 2,
  kCPUPinned = 3
};

/*!
* \brief Context interface
*/
class Context {
 public:
  /*!
  * \brief Context constructor
  * \param type type of the device
  * \param id id of the device
  */
  Context(const DeviceType &type, int id) : type_(type), id_(id) {}
  /*!
  * \return the type of the device
  */
  DeviceType GetDeviceType() const { return type_; }
  /*!
  * \return the id of the device
  */
  int GetDeviceId() const { return id_; }

  /*!
   * \brief Return a GPU context
   * \param device_id id of the device
   * \return the corresponding GPU context
   */
  static Context gpu(int device_id = 0) {
    return Context(DeviceType::kGPU, device_id);
  }

  /*!
   * \brief Return a CPU context
   * \param device_id id of the device. this is not needed by CPU
   * \return the corresponding CPU context
   */
  static Context cpu(int device_id = 0) {
    return Context(DeviceType::kCPU, device_id);
  }

 private:
  DeviceType type_;
  int id_;
};

/*!
* \brief struct to store NDArrayHandle
*/
struct NDBlob {
 public:
  /*!
  * \brief default constructor
  */
  NDBlob() : handle_(nullptr) {}
  /*!
  * \brief construct with a NDArrayHandle
  * \param handle NDArrayHandle to store
  */
  explicit NDBlob(NDArrayHandle handle) : handle_(handle) {}
  /*!
  * \brief destructor, free the NDArrayHandle
  */
  ~NDBlob() { MXNDArrayFree(handle_); }
  /*!
  * \brief the NDArrayHandle
  */
  NDArrayHandle handle_;

 private:
  NDBlob(const NDBlob &);
  NDBlob &operator=(const NDBlob &);
};

/*!
* \brief NDArray interface
*/
class NDArray {
 public:
  /*!
  * \brief construct with a none handle
  */
  NDArray();
  /*!
  * \brief construct with a NDArrayHandle
  */
  explicit NDArray(const NDArrayHandle &handle);
  /*!
  * \brief construct a new dynamic NDArray
  * \param shape the shape of array
  * \param context context of NDArray
  * \param delay_alloc whether delay the allocation
  * \param dtype data type of NDArray
  */
  NDArray(const std::vector<mx_uint> &shape, const Context &context,
          bool delay_alloc = true, int dtype = 0);
  /*!
  * \brief construct a new dynamic NDArray
  * \param shape the shape of array
  * \param constext context of NDArray
  * \param delay_alloc whether delay the allocation
  * \param dtype data type of NDArray
  */
  NDArray(const Shape &shape, const Context &context,
          bool delay_alloc = true, int dtype = 0);
  NDArray(const mx_float *data, size_t size);
  /*!
  * \brief construct a new dynamic NDArray
  * \param data the data to create NDArray from
  * \param shape the shape of array
  * \param constext context of NDArray
  */
  NDArray(const mx_float *data, const Shape &shape, const Context &context);
  /*!
  * \brief construct a new dynamic NDArray
  * \param data the data to create NDArray from
  * \param shape the shape of array
  * \param constext context of NDArray
  */
  NDArray(const std::vector<mx_float> &data, const Shape &shape,
          const Context &context);
  explicit NDArray(const std::vector<mx_float> &data);
  NDArray operator+(mx_float scalar);
  NDArray operator-(mx_float scalar);
  NDArray operator*(mx_float scalar);
  NDArray operator/(mx_float scalar);
  NDArray operator%(mx_float scalar);
  NDArray operator+(const NDArray &);
  NDArray operator-(const NDArray &);
  NDArray operator*(const NDArray &);
  NDArray operator/(const NDArray &);
  NDArray operator%(const NDArray &);
  /*!
  * \brief set all the elements in ndarray to be scalar
  * \param scalar the scalar to set
  * \return reference of self
  */
  NDArray &operator=(mx_float scalar);
  /*!
  * \brief elementwise add to current space
  *  this mutate the current NDArray
  * \param scalar the data to add
  * \return reference of self
  */
  NDArray &operator+=(mx_float scalar);
  /*!
  * \brief elementwise subtract from current ndarray
  * this mutate the current NDArray
  * \param scalar the data to subtract
  * \return reference of self
  */
  NDArray &operator-=(mx_float scalar);
  /*!
  * \brief elementwise multiplication to current ndarray
  *  this mutate the current NDArray
  * \param scalar the data to subtract
  * \return reference of self
  */
  NDArray &operator*=(mx_float scalar);
  /*!
  * \brief elementwise division from current ndarray
  *  this mutate the current NDArray
  * \param scalar the data to subtract
  * \return reference of self
  */
  NDArray &operator/=(mx_float scalar);
  /*!
  * \brief elementwise modulo from current ndarray
  *  this mutate the current NDArray
  * \param scalar the data to subtract
  * \return reference of self
  */
  NDArray &operator%=(mx_float scalar);
  /*!
  * \brief elementwise add to current space
  *  this mutate the current NDArray
  * \param src the data to add
  * \return reference of self
  */
  NDArray &operator+=(const NDArray &src);
  /*!
  * \brief elementwise subtract from current ndarray
  * this mutate the current NDArray
  * \param src the data to subtract
  * \return reference of self
  */
  NDArray &operator-=(const NDArray &src);
  /*!
  * \brief elementwise multiplication to current ndarray
  *  this mutate the current NDArray
  * \param src the data to subtract
  * \return reference of self
  */
  NDArray &operator*=(const NDArray &src);
  /*!
  * \brief elementwise division from current ndarray
  *  this mutate the current NDArray
  * \param src the data to subtract
  * \return reference of self
  */
  NDArray &operator/=(const NDArray &src);
  /*!
  * \brief elementwise modulo from current ndarray
  *  this mutate the current NDArray
  * \param src the data to subtract
  * \return reference of self
  */
  NDArray &operator%=(const NDArray &src);
  NDArray ArgmaxChannel();
  /*!
  * \brief Do a synchronize copy from a contiguous CPU memory region.
  *
  *  This function will call WaitToWrite before the copy is performed.
  *  This is useful to copy data from existing memory region that are
  *  not wrapped by NDArray(thus dependency not being tracked).
  *
  * \param data the data source to copy from.
  * \param size the memory size we want to copy from.
  */
  void SyncCopyFromCPU(const mx_float *data, size_t size);
  /*!
  * \brief Do a synchronize copy from a contiguous CPU memory region.
  *
  *  This function will call WaitToWrite before the copy is performed.
  *  This is useful to copy data from existing memory region that are
  *  not wrapped by NDArray(thus dependency not being tracked).
  *
  * \param data the data source to copy from, int the form of mx_float vector
  */
  void SyncCopyFromCPU(const std::vector<mx_float> &data);
  /*!
  * \brief Do a synchronize copy to a contiguous CPU memory region.
  *
  *  This function will call WaitToRead before the copy is performed.
  *  This is useful to copy data from existing memory region that are
  *  not wrapped by NDArray(thus dependency not being tracked).
  *
  * \param data the data source to copyinto.
  * \param size the memory size we want to copy into. Defualt value is Size()
  */
  void SyncCopyToCPU(mx_float *data, size_t size = 0);
  /*!
  * \brief Do a synchronize copy to a contiguous CPU memory region.
  *
  *  This function will call WaitToRead before the copy is performed.
  *  This is useful to copy data from existing memory region that are
  *  not wrapped by NDArray(thus dependency not being tracked).
  *
  * \param data the data source to copyinto.
  * \param size the memory size we want to copy into. Defualt value is Size()
  */
  void SyncCopyToCPU(std::vector<mx_float> *data, size_t size = 0);
  /*!
  * \brief copy the content of current array to a target array.
  * \param other the target NDArray
  * \return the target NDarray
  */
  NDArray CopyTo(NDArray * other) const;
  /*!
  * \brief return a new copy to this NDArray
  * \param Context the new context of this NDArray
  * \return the new copy
  */
  NDArray Copy(const Context &) const;
  /*!
  * \brief return offset of the element at (h, w)
  * \param h height position
  * \param w width position
  * \return offset of two dimensions array
  */
  size_t Offset(size_t h = 0, size_t w = 0) const;
  /*!
   * \brief return offset of three dimensions array
   * \param c channel position
   * \param h height position
   * \param w width position
   * \return offset of three dimensions array
   */
  size_t Offset(size_t c, size_t h, size_t w) const;
  /*!
  * \brief return value of the element at (index)
  * \param index  position
  * \return value of one dimensions array
  */
  mx_float At(size_t index) const;
  /*!
  * \brief return value of the element at (h, w)
  * \param h height position
  * \param w width position
  * \return value of two dimensions array
  */
  mx_float At(size_t h, size_t w) const;
  /*!
   * \brief return value of three dimensions array
   * \param c channel position
   * \param h height position
   * \param w width position
   * \return value of three dimensions array
   */
  mx_float At(size_t c, size_t h, size_t w) const;
  /*!
  * \brief Slice a NDArray
  * \param begin begin index in first dim
  * \param end end index in first dim
  * \return sliced NDArray
  */
  NDArray Slice(mx_uint begin, mx_uint end) const;
  /*!
  * \brief Return a reshaped NDArray that shares memory with current one
  * \param new_shape the new shape
  * \return reshaped NDarray
  */
  NDArray Reshape(const Shape &new_shape) const;
  /*!
  * \brief Block until all the pending write operations with respect
  *    to current NDArray are finished, and read can be performed.
  */
  void WaitToRead() const;
  /*!
  * \brief Block until all the pending read/write operations with respect
  *    to current NDArray are finished, and write can be performed.
  */
  void WaitToWrite();
  /*!
  * \brief Block until all the pending read/write operations with respect
  *    to current NDArray are finished, and read/write can be performed.
  */
  static void WaitAll();
  /*!
  * \brief Sample gaussian distribution for each elements of out.
  * \param mu mean of gaussian distribution.
  * \param sigma standard deviation of gaussian distribution.
  * \param out output NDArray.
  */
  static void SampleGaussian(mx_float mu, mx_float sigma, NDArray *out);
  /*!
  * \brief Sample uniform distribution for each elements of out.
  * \param begin lower bound of distribution.
  * \param end upper bound of distribution.
  * \param out output NDArray.
  */
  static void SampleUniform(mx_float begin, mx_float end, NDArray *out);
  /*!
  * \brief Load NDArrays from binary file.
  * \param file_name name of the binary file.
  * \param array_list a list of NDArrays returned, do not fill the list if
  * nullptr is given.
  * \param array_map a map from names to NDArrays returned, do not fill the map
  * if nullptr is given or no names is stored in binary file.
  */
  static void Load(const std::string &file_name,
                   std::vector<NDArray> *array_list = nullptr,
                   std::map<std::string, NDArray> *array_map = nullptr);
  /*!
  * \brief Load map of NDArrays from binary file.
  * \param file_name name of the binary file.
  * \return a list of NDArrays.
  */
  static std::map<std::string, NDArray> LoadToMap(const std::string &file_name);
  /*!
  * \brief Load list of NDArrays from binary file.
  * \param file_name name of the binary file.
  * \return a map from names to NDArrays.
  */
  static std::vector<NDArray> LoadToList(const std::string &file_name);
  /*!
  * \brief Load NDArrays from buffer.
  * \param buffer Pointer to buffer. (ie contents of param file)
  * \param size Size of buffer
  * \param array_list a list of NDArrays returned, do not fill the list if
  * nullptr is given.
  * \param array_map a map from names to NDArrays returned, do not fill the map
  * if nullptr is given or no names is stored in binary file.
  */
  static void LoadFromBuffer(const void *buffer, size_t size,
                   std::vector<NDArray> *array_list = nullptr,
                   std::map<std::string, NDArray> *array_map = nullptr);
  /*!
  * \brief Load map of NDArrays from buffer.
  * \param buffer Pointer to buffer. (ie contents of param file)
  * \param size Size of buffer
  * \return a list of NDArrays.
  */
  static std::map<std::string, NDArray> LoadFromBufferToMap(const void *buffer, size_t size);
  /*!
  * \brief Load list of NDArrays from buffer.
  * \param buffer Pointer to buffer. (ie contents of param file)
  * \param size Size of buffer
  * \return a map from names to NDArrays.
  */
  static std::vector<NDArray> LoadFromBufferToList(const void *buffer, size_t size);
  /*!
  * \brief save a map of string->NDArray to binary file.
  * \param file_name name of the binary file.
  * \param array_map a map from names to NDArrays.
  */
  static void Save(const std::string &file_name,
                   const std::map<std::string, NDArray> &array_map);
  /*!
  * \brief save a list of NDArrays to binary file.
  * \param file_name name of the binary file.
  * \param array_list a list of NDArrays.
  */
  static void Save(const std::string &file_name,
                   const std::vector<NDArray> &array_list);
  /*!
  * \return the size of current NDArray, a.k.a. the production of all shape dims
  */
  size_t Size() const;
  /*!
  * \return the shape of current NDArray, in the form of mx_uint vector
  */
  std::vector<mx_uint> GetShape() const;
  /*!
  * \return the data type of current NDArray
  */
  int GetDType() const;
  /*!
  * \brief Get the pointer to data (IMPORTANT: The ndarray should not be in GPU)
  * \return the data pointer to the current NDArray
  */
  const mx_float *GetData() const;

  /*!
  * \return the context of NDArray
  */
  Context GetContext() const;

  /*!
  * \return the NDArrayHandle of the current NDArray
  */
  NDArrayHandle GetHandle() const { return blob_ptr_->handle_; }

 private:
  std::shared_ptr<NDBlob> blob_ptr_;
};

std::ostream& operator<<(std::ostream& out, const NDArray &ndarray);
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_NDARRAY_H_
