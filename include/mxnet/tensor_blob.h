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
 *  Copyright (c) 2014 by Contributors
 * \file tensor_blob.h
 * \brief TBlob class that holds common representation of
 *  arbirary dimension tensor, can be used to transformed
 *  to normal fixed dimenson tensor
 * \author Tianqi Chen
 */
#ifndef MXNET_TENSOR_BLOB_H_
#define MXNET_TENSOR_BLOB_H_

#include <dmlc/logging.h>
#include <dmlc/json.h>
#include <dlpack/dlpack.h>
#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include "./base.h"

namespace mxnet {

// redefine DLPack enumeration to be backward compatible.
constexpr const int kCPU = kDLCPU;
constexpr const int kGPU = kDLGPU;
// extension type code under TVM function.
// Currently NNVM reserved 16 to 19 type code from TVM
// 16, 17, 18 is used by NNVM compiler already.
// Pick code 19 for MXNet NDArray
constexpr const int kTVMNDArrayTypeCode = 19;

/* Forward declaration for friend declaration in TBlob */
class NDArray;

/*!
 * \brief tensor blob class that can be used to hold tensor of any dimension,
 *  any device and any data type,
 *  This is a weak type that can be used to transfer data through interface
 *  TBlob itself doesn't involve any arithmetic operations,
 *  but it can be converted to tensor of fixed dimension for further operations
 *
 *  Like tensor, this data structure is like a pointer class and do not
 *  implicit allocated, de-allocate space.
 *  This data structure can be helpful to hold tensors of different dimensions
 *  and wait for further processing
 */
class TBlob {
  friend class NDArray;
 public:
  /*! \brief pointer to the data */
  void *dptr_;
  /*! \brief shape of the tensor */
  mxnet::TShape shape_;
  /*! \brief type flag of the tensor blob */
  int type_flag_;

  /*! \brief default constructor, default copy assign will work */
  TBlob(void)
      : dptr_(nullptr),
        type_flag_(mshadow::DataType<real_t>::kFlag) {
    SetDLTensor(cpu::kDevMask, 0);
  }
  /*!
   * \brief constructor that construct TBlob from contiguous memory
   * \param dptr the pointer to the memory
   * \param shape the shape of the data
   * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
   * \param dev_id the device id
   */
  template<typename DType>
  TBlob(DType *dptr, const mxnet::TShape &shape, int dev_mask, int dev_id = -1)
      : dptr_(dptr), shape_(shape),
        type_flag_(mshadow::DataType<DType>::kFlag) {
    SetDLTensor(dev_mask, dev_id);
  }
  /*!
   * \brief constructor that construct TBlob from contiguous memory
   * \param dptr the pointer to the memory
   * \param shape the shape of the data
   * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
   * \param type_flag the type flag. Can be one of enum mshadow::dtype
   * \param dev_id the device id
   */
  TBlob(void *dptr, const mxnet::TShape &shape, int dev_mask, int type_flag, int dev_id = -1)
      : dptr_(dptr), shape_(shape), type_flag_(type_flag) {
    SetDLTensor(dev_mask, dev_id);
  }
  /*!
   * \brief constructor that construct TBlob from DLTensor
   * \param DLTensor Object
   */
  explicit TBlob(const DLTensor &dltensor)
      : dptr_(dltensor.data),
        shape_(mxnet::TShape(dltensor.shape, dltensor.shape + dltensor.ndim)),
        type_flag_(DLDataTypeTransform(dltensor.dtype)),
        dltensor_(dltensor) {
    // compactness check for DLTensor
    if (dltensor.strides != nullptr) {
      // check strides
      const int &ndim = dltensor.ndim;
      const int64_t *shape = dltensor.shape;
      const int64_t *strides = dltensor.strides;
      if (ndim >= 1) {
        bool err = false;
        if (strides[ndim - 1] != 1) {
          err = true;
        } else {
          for (int i = ndim - 2; i >= 0; --i) {
            if (strides[i] != shape[i + 1] * strides[i + 1]) {
              err = true;
              break;
            }
          }
        }
        if (err) {
          LOG(FATAL) << "Unsupported DLPack because MXNet only support compact tensor now";
        }
      }
    }
  }
  /*!
   * \brief constructor from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  TBlob(const mshadow::Tensor<Device, dim, DType> &src) {  // NOLINT(*)
    *this = src;
  }
  /*!
   * \brief constructor from TBlob (copy constructor)
   * \param src source TBlob
   */
  TBlob(const TBlob &src): dptr_(src.dptr_), shape_(src.shape_), type_flag_(src.type_flag_) {
    this->SetDLTensor(src.dev_mask(), src.dev_id());
  }
  /*!
   * \brief assignment from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   * \return reference of self
   */
  template<typename Device, int dim, typename DType>
  inline TBlob &operator=(const mshadow::Tensor<Device, dim, DType> &src) {
    dptr_ = src.dptr_;
    shape_ = src.shape_;
    type_flag_ = mshadow::DataType<DType>::kFlag;
    SetDLTensor(Device::kDevMask, -1);
    return *this;
  }
  /*!
   * \brief assignment from TBlob (copy assignment)
   * \param src source TBlob
   * \return reference of self
   */
  inline TBlob &operator=(const TBlob &src) {
    dptr_ = src.dptr_;
    shape_ = src.shape_;
    type_flag_ = src.type_flag_;
    SetDLTensor(src.dev_mask(), src.dev_id());
    return *this;
  }
  /*!
   * \return whether the tensor's memory is continuous
   */
  inline bool CheckContiguous(void) const {
    return true;
  }
  /*!
   * \brief reshape to shape
   * \param shape desired shape
   * \return reshaped blob
   */
  inline TBlob reshape(const mxnet::TShape& shape) const {
    CHECK_EQ(this->shape_.Size(), shape.Size()) << "Shape size mismatch "
    << this->shape_.Size() << " v.s. "  << shape.Size();
    TBlob ret(this->dptr_, shape, this->dev_mask(), this->type_flag_, this->dev_id());
    return ret;
  }
  /*!
   * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, typename DType>
  inline mshadow::Tensor<Device, 2, DType> FlatTo2D(
    mshadow::Stream<Device> *stream = nullptr) const {
    CHECK(Device::kDevMask == this->dev_mask())
      << "TBlob.get: device type do not match specified type";
    CHECK(mshadow::DataType<DType>::kFlag == type_flag_)
      << "TBlob.get_with_shape: data type do not match specified type."
      << "Expected: " << mshadow::dtype_string(type_flag_)
      << " v.s. given " << mshadow::dtype_string(mshadow::DataType<DType>::kFlag);
    return mshadow::Tensor<Device, 2, DType>(static_cast<DType*>(dptr_),
                                             shape_.FlatTo2D(),
                                             stream);
  }
  /*!
   * \brief flatten the tensor to 1 dimension, collapse all the dimensions together.
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, typename DType>
  inline mshadow::Tensor<Device, 1, DType> FlatTo1D(
      mshadow::Stream<Device> *stream = nullptr) const {
    return this->get_with_shape<Device, 1, DType>(
        mshadow::Shape1(shape_.Size()), stream);
  }
  /*! \brief return number of dimension of the tensor inside */
  inline int ndim(void) const {
    return shape_.ndim();
  }
  /*!
   * \brief return size of i-th dimension, start counting from highest dimension.
   * return type needs to be a signed integer.
   * \param idx the dimension count from the highest dimensin
   * \return the size. -1 means unknown size to support zero-size tensor.
   */
  inline index_t size(index_t idx) const {
    return shape_[idx];
  }
  /*! \brief total number of elements in the tensor */
  inline size_t Size(void) const {
    return shape_.Size();
  }
  /*! \brief get pointer in dtype */
  template<typename DType>
  inline DType* dptr() const {
    CHECK(mshadow::DataType<DType>::kFlag == type_flag_)
      << "TBlob.get_with_shape: data type do not match specified type."
      << "Expected: " << mshadow::dtype_string(type_flag_)
      << " v.s. given " << mshadow::dtype_string(mshadow::DataType<DType>::kFlag);
    return static_cast<DType*>(dptr_);
  }
  /*! \brief device mask of the corresponding device */
  inline int dev_mask() const {
    return dltensor_.ctx.device_type;
  }
  /*! \brief device index of the corresponding device */
  inline int dev_id() const {
    return dltensor_.ctx.device_id;
  }
  /*!
   * \brief return the corresponding DLTensor
   * \return the address of internal DLTensor
   */
  inline const DLTensor& dltensor() const {
    return dltensor_;
  }

  /*!
   * \brief fetch the tensor, with respect to specific dimension
   * if dim do not match the stored dimension, an error will be issued
   * \return the tensor requested
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim dimension of the tensor
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline mshadow::Tensor<Device, dim, DType> get(mshadow::Stream<Device> *stream = nullptr) const {
    CHECK(Device::kDevMask == this->dev_mask())
      << "TBlob.get: device type do not match specified type";
    return mshadow::Tensor<Device, dim, DType>(dptr<DType>(),
        shape_.get<dim>(), shape_[shape_.ndim() - 1], stream);
  }
  /*!
   * \brief fetch a tensor in given shape
   *  If size do not match the stored size, an error will be issued
   * \return the tensor requested
   * \param shape the shape required
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim dimension of the tensor
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline mshadow::Tensor<Device, dim, DType> get_with_shape(
      const mshadow::Shape<dim> &shape,
      mshadow::Stream<Device> *stream = nullptr) const {
    CHECK(Device::kDevMask == this->dev_mask())
      << "TBlob.get: device type do not match specified type";
    CHECK_EQ(this->CheckContiguous(), true) << "TBlob.get_reshape: must be contiguous";
    CHECK_EQ(this->shape_.Size(), static_cast<size_t>(shape.Size()))
      << "TBlob.get_with_shape: new and old shape do not match total elements";
    return mshadow::Tensor<Device, dim, DType>(dptr<DType>(), shape,
                                               shape[dim - 1], stream);
  }
  /*!
   * \brief flatten the tensor to 3 dimension,
   *  collapse the dimension before and after specified axis.
   * \param axis The axis specified.
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, typename DType>
  inline mshadow::Tensor<Device, 3, DType> FlatTo3D(
      int axis, mshadow::Stream<Device> *stream = nullptr) const {
    return this->get_with_shape<Device, 3, DType>(
        this->shape_.FlatTo3D(axis), stream);
  }
  /*!
   * \brief flatten the tensor to 3 dimension,
   *  collapse the dimension: [0, axis_begin), [axis_begin, axis_end], (axis_end, ndim).
   * \param axis_begin The beginning axis specified.
   * \param axis_end The ending axis specified.
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, typename DType>
  inline mshadow::Tensor<Device, 3, DType> FlatTo3D(
      int axis_begin, int axis_end,
      mshadow::Stream<Device> *stream = nullptr) const {
    return this->get_with_shape<Device, 3, DType>(
        this->shape_.FlatTo3D(axis_begin, axis_end), stream);
  }
  /*!
   * \brief flatten the tensor to specified number of dimensions,
   *  collapse the highest dimensions or pad with higher dimensions
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim desired number of dimensions of returned tensor
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */
  template<typename Device, int dim, typename DType>
  inline mshadow::Tensor<Device, dim, DType> FlatToKD(
     mshadow::Stream<Device> *stream = nullptr) const {
    mshadow::Shape<dim> shape;
    shape[0] = 1;
    // Pad higher dimensions in case dim > ndim()
    for (int i = 0; i < dim - ndim(); ++i) {
      shape[i] = 1;
    }
    // Collapse higher dimensions in case dim < ndim()
    for (int i = 0; i < ndim() - dim + 1; ++i) {
      shape[0] *= shape_[i];
    }
    // Preserve lower dimensions.
    for (int i = std::max(0, ndim() - dim + 1); i < ndim(); ++i) {
      shape[i - ndim() + dim] = shape_[i];
    }
    return this->get_with_shape<Device, dim, DType>(shape, stream);
  }

 private:
  static DLDataType DTypeTransform(int type_flag) {
    switch (type_flag) {
      case mshadow::kFloat32: return DLDataType{kDLFloat, 32, 1};
      case mshadow::kFloat64: return DLDataType{kDLFloat, 64, 1};
      case mshadow::kFloat16: return DLDataType{kDLFloat, 16, 1};
      case mshadow::kBfloat16: return DLDataType{kDLBfloat, 16, 1};
      case mshadow::kUint8: return DLDataType{kDLUInt, 8, 1};
      case mshadow::kInt32: return DLDataType{kDLInt, 32, 1};
      case mshadow::kInt8: return DLDataType{kDLInt, 8, 1};
      case mshadow::kUint16: return DLDataType{kDLUInt, 16, 1};
      case mshadow::kInt16: return DLDataType{kDLInt, 16, 1};
      case mshadow::kInt64: return DLDataType{kDLInt, 64, 1};
      case mshadow::kBool: return DLDataType{kDLUInt, 1, 1};
      default: {
        LOG(FATAL) << "Unknown type_flag=" << type_flag;
        return DLDataType();
      }
    }
  }
  static int DLDataTypeTransform(DLDataType dldata_type) {
    if (dldata_type.lanes != 1) {
      LOG(FATAL) << "Unsupported DLDataType whose lanes != 1";
    }
    switch (dldata_type.code) {
      case kDLFloat:
        switch (dldata_type.bits) {
          case 16: return mshadow::kFloat16;
          case 32: return mshadow::kFloat32;
          case 64: return mshadow::kFloat64;
        }
        break;
      case kDLBfloat:
        switch (dldata_type.bits) {
          case 16: return mshadow::kBfloat16;
        }
        break;
      case kDLUInt:
        switch (dldata_type.bits) {
          case 1: return mshadow::kBool;
          case 8: return mshadow::kUint8;
          case 16: return mshadow::kUint16;
        }
        break;
      case kDLInt:
        switch (dldata_type.bits) {
          case 8: return mshadow::kInt8;
          case 16: return mshadow::kInt16;
          case 32: return mshadow::kInt32;
          case 64: return mshadow::kInt64;
        }
        break;
    }
    LOG(FATAL) << "Unknown DLDataType{" << dldata_type.code
               << ", " << dldata_type.bits
               << ", " << dldata_type.lanes << "}";
    return mshadow::kFloat32;
  }

  inline void SetDLTensor(int dev_mask, int dev_id) {
    dltensor_.data = dptr_;
    dltensor_.ctx = DLContext{static_cast<DLDeviceType>(dev_mask), dev_id};
    dltensor_.ndim = shape_.ndim();
    dltensor_.dtype = DTypeTransform(type_flag_);
    dltensor_.shape = shape_.data();
    dltensor_.strides = nullptr;
    dltensor_.byte_offset = 0;
  }

 private:
  /*! \brief corresponding DLTensor of this TBlob */
  DLTensor dltensor_;
};
}  // namespace mxnet

namespace dmlc {
// Add a few patches to support mxnet::TShape in dmlc/parameter.
DMLC_DECLARE_TYPE_NAME(mxnet::TShape, "Shape(tuple)");
DMLC_DECLARE_TYPE_NAME(mxnet::Tuple<int>, "Shape(tuple)");
DMLC_DECLARE_TYPE_NAME(mxnet::Tuple<dmlc::optional<int>>, "Shape(tuple)");
DMLC_DECLARE_TYPE_NAME(nnvm::Tuple<int>, "Shape(tuple)");
DMLC_DECLARE_TYPE_NAME(nnvm::Tuple<dmlc::optional<int>>, "Shape(tuple)");

namespace parameter {

template<>
class FieldEntry<mxnet::TShape>
    : public FieldEntryBase<FieldEntry<mxnet::TShape>, mxnet::TShape> {
 public:
  FieldEntry() : enforce_nonzero_(false), expect_ndim_(0) {}
  // parent class
  typedef FieldEntryBase<FieldEntry<mxnet::TShape>, mxnet::TShape> Parent;

  virtual void Check(void *head) const {
    Parent::Check(head);
    mxnet::TShape &v = this->Get(head);
    if (expect_ndim_ != 0 && v.ndim() != expect_ndim_) {
      std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " has wrong dimensions, expected dimension=" << expect_ndim_;
        throw dmlc::ParamError(os.str());
    }
    if (enforce_nonzero_) {
      for (int i = 0; i < v.ndim(); ++i) {
        if (v[i] == 0U) {
          std::ostringstream os;
          os << "value " << v << "for Parameter " << this->key_
             << " is invalid, the input shape must be nonzero in all dimensions";
          throw dmlc::ParamError(os.str());
        }
      }
    }
  }
  inline FieldEntry<mxnet::TShape> &enforce_nonzero() {
    this->enforce_nonzero_ = true;
    return this->self();
  }
  inline FieldEntry<mxnet::TShape> &set_expect_ndim(int ndim) {
    expect_ndim_ = ndim;
    return this->self();
  }

 private:
  // whether all the entries need to be nonzero
  bool enforce_nonzero_;
  // expected number of dimension, default = 0 means no restriction.
  int expect_ndim_;
};

}  // namespace parameter
}  // namespace dmlc

#endif  // MXNET_TENSOR_BLOB_H_
