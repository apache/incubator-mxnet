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
* \file shape.h
* \brief definition of shape
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_CPP_SHAPE_H_
#define MXNET_CPP_SHAPE_H_

#include <istream>
#include <ostream>
#include <algorithm>
#include <vector>
#include "mxnet-cpp/base.h"

namespace mxnet {
namespace cpp {

/*!
* \brief dynamic shape class that can hold shape
*   of arbirary dimension
*/
struct Shape {
 public:
  /*! \brief constructor */
  Shape()
    : ndim_(0),
    num_heap_allocated_(0),
    data_heap_(nullptr) {}
  /*!
  * \brief constructor from a vector of index_t
  * \param v the vector
  */
  explicit Shape(const std::vector<index_t> &v)
    : ndim_(v.size()) {
    if (ndim_ <= kStackCache) {
      data_heap_ = nullptr;
      num_heap_allocated_ = 0;
      std::copy(v.begin(), v.end(), data_stack_);
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      std::copy(v.begin(), v.end(), data_heap_);
    }
  }
  /*!
  * \brief constructor one dimmension shape
  * \param s1 size of the first dimmension
  */
  explicit Shape(index_t s1)
    : ndim_(1) {
    if (ndim_ <= kStackCache) {
      data_heap_ = nullptr;
      num_heap_allocated_ = 0;
      data_stack_[0] = s1;
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      data_heap_[0] = s1;
    }
  }
  /*!
  * \brief constructor two dimmension shape
  * \param s1 size of the first dimmension
  * \param s2 size of the second dimmension
  */
  Shape(index_t s1, index_t s2)
    : ndim_(2) {
    if (ndim_ <= kStackCache) {
      data_heap_ = nullptr;
      num_heap_allocated_ = 0;
      data_stack_[0] = s1;
      data_stack_[1] = s2;
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      data_heap_[0] = s1;
      data_heap_[1] = s2;
    }
  }
  /*!
  * \brief constructor three dimmension shape
  * \param s1 size of the first dimmension
  * \param s2 size of the second dimmension
  * \param s3 size of the third dimmension
  */
  Shape(index_t s1, index_t s2, index_t s3)
    : ndim_(3) {
    if (ndim_ <= kStackCache) {
      data_heap_ = nullptr;
      num_heap_allocated_ = 0;
      data_stack_[0] = s1;
      data_stack_[1] = s2;
      data_stack_[2] = s3;
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      data_heap_[0] = s1;
      data_heap_[1] = s2;
      data_heap_[2] = s3;
    }
  }
  /*!
  * \brief constructor four dimmension shape
  * \param s1 size of the first dimmension
  * \param s2 size of the second dimmension
  * \param s3 size of the third dimmension
  * \param s4 size of the fourth dimmension
  */
  Shape(index_t s1, index_t s2, index_t s3, index_t s4)
    : ndim_(4) {
    if (ndim_ <= kStackCache) {
      data_heap_ = nullptr;
      num_heap_allocated_ = 0;
      data_stack_[0] = s1;
      data_stack_[1] = s2;
      data_stack_[2] = s3;
      data_stack_[3] = s4;
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      data_heap_[0] = s1;
      data_heap_[1] = s2;
      data_heap_[2] = s3;
      data_heap_[3] = s4;
    }
  }
  /*!
  * \brief constructor five dimmension shape
  * \param s1 size of the first dimmension
  * \param s2 size of the second dimmension
  * \param s3 size of the third dimmension
  * \param s4 size of the fourth dimmension
  * \param s5 size of the fifth dimmension
  */
  Shape(index_t s1, index_t s2, index_t s3, index_t s4, index_t s5)
    : ndim_(5) {
    if (ndim_ <= kStackCache) {
      data_heap_ = nullptr;
      num_heap_allocated_ = 0;
      data_stack_[0] = s1;
      data_stack_[1] = s2;
      data_stack_[2] = s3;
      data_stack_[3] = s4;
      data_stack_[4] = s5;
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      data_heap_[0] = s1;
      data_heap_[1] = s2;
      data_heap_[2] = s3;
      data_heap_[3] = s4;
      data_heap_[4] = s5;
    }
  }
  /*!
  * \brief constructor from Shape
  * \param s the source shape
  */
  Shape(const Shape &s)
    : ndim_(s.ndim_) {
    if (ndim_ <= kStackCache) {
      data_heap_ = nullptr;
      num_heap_allocated_ = 0;
      std::copy(s.data_stack_, s.data_stack_ + ndim_, data_stack_);
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      std::copy(s.data_heap_, s.data_heap_ + ndim_, data_heap_);
    }
  }
#if MSHADOW_IN_CXX11
  /*!
  * \brief move constructor from Shape
  * \param s the source shape
  */
  Shape(Shape &&s)
    : ndim_(s.ndim_),
    num_heap_allocated_(s.num_heap_allocated_),
    data_heap_(s.data_heap_) {
    if (ndim_ <= kStackCache) {
      std::copy(s.data_stack_, s.data_stack_ + ndim_, data_stack_);
    }
    // remove data heap space from s
    s.data_heap_ = nullptr;
  }
#endif
  /*! \brief destructor */
  ~Shape() {
    // data_heap_ can be nullptr
    delete[] data_heap_;
  }
  /*!
  * \brief copy shape from content betwen two iterators
  * \param begin the beginning of iterator
  * \param end the end of the iterator
  * \tparam RandomAccessIterator iterator type
  */
  template<typename RandomAccessIterator>
  inline void CopyFrom(RandomAccessIterator begin,
    RandomAccessIterator end) {
    this->SetDim(end - begin);
    std::copy(begin, end, data());
  }
  /*!
  * \brief assignment from shape
  * \param shape source shape
  * \return reference of self
  */
  inline Shape &operator=(const Shape &shape) {
    this->SetDim(shape.ndim_);
    const index_t *src = shape.data();
    std::copy(src, src + ndim_, data());
    return *this;
  }
  /*!
  * \brief assignment from vector
  * \param shape source shape
  * \return reference of self
  */
  inline Shape &operator=(const std::vector<index_t> &shape) {
    this->CopyFrom(shape.begin(), shape.end());
    return *this;
  }
  /*! \return the data content of the shape */
  inline const index_t *data() const {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \return the data content of the shape */
  inline index_t *data() {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \brief return number of dimension of the tensor inside */
  inline index_t ndim(void) const {
    return ndim_;
  }
  /*!
  * \brief get corresponding index
  * \param i dimension index
  * \return the corresponding dimension size
  */
  inline index_t &operator[](index_t i) {
    return data()[i];
  }
  /*!
  * \brief get corresponding index
  * \param i dimension index
  * \return the corresponding dimension size
  */
  inline const index_t &operator[](index_t i) const {
    return data()[i];
  }
  /*! \brief total number of elements in the tensor */
  inline size_t Size(void) const {
    size_t size = 1;
    const index_t *d = this->data();
    for (index_t i = 0; i < ndim_; ++i) {
      size *= d[i];
    }
    return size;
  }
  /*!
  * \return whether two shape equals
  * \param s the shape to compare against
  */
  inline bool operator==(const Shape &s) const {
    if (ndim_ != s.ndim_) return false;
    if (ndim_ <= kStackCache) {
      for (index_t i = 0; i < ndim_; ++i) {
        if (data_stack_[i] != s.data_stack_[i]) return false;
      }
    } else {
      for (index_t i = 0; i < ndim_; ++i) {
        if (data_heap_[i] != s.data_heap_[i]) return false;
      }
    }
    return true;
  }
  /*!
  * \return whether two shape not equals
  * \param s the shape to compare against
  */
  inline bool operator!=(const Shape &s) const {
    return !(*this == s);
  }

  friend std::ostream &operator<<(std::ostream &os, const Shape &shape);
  friend std::istream &operator>>(std::istream &is, Shape &shape);

 private:
  // the shape will be stored in data_stack_
  // when dimension is smaller than kStackCache
  // when it is bigger, it will be stored in data_heap_;
  /*! \brief size of in stack space */
  static const index_t kStackCache = 5;
  /*! \brief number of dimnsion of the shape */
  index_t ndim_;
  /*! \brief number of cells allocated in data_heap_ */
  index_t num_heap_allocated_;
  /*! \brief in stack space used to store shape when it is small */
  index_t data_stack_[kStackCache];
  /*! \brief space to store shape when dimension is big*/
  index_t *data_heap_;
  /*!
  * \brief internal function to set the dimension
  * \param dim the dimension of the shape
  */
  inline void SetDim(index_t dim) {
    if (dim > kStackCache &&
      dim > num_heap_allocated_) {
      // data_heap_ can be nullptr
      delete[] data_heap_;
      data_heap_ = new index_t[dim];
      num_heap_allocated_ = dim;
    }
    ndim_ = dim;
  }
};

/*!
* \brief allow string printing of the shape
* \param os the output stream
* \param shape the shape
* \return the ostream
*/
inline std::ostream &operator<<(std::ostream &os, const Shape &shape) {
  os << '(';
  for (index_t i = 0; i < shape.ndim(); ++i) {
    if (i != 0) os << ',';
    os << static_cast<int>(shape[i]);  // Supports negative Shape 'special codes' for inferring
  }
  // python style tuple
  if (shape.ndim() == 1) os << ',';
  os << ')';
  return os;
}

/*!
* \brief read shape from the istream
* \param is the input stream
* \param shape the shape
* \return the istream
*/
inline std::istream &operator>>(std::istream &is, Shape &shape) {
  // get (
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  index_t idx;
  std::vector<index_t> tmp;
  while (is >> idx) {
    tmp.push_back(idx);
    char ch;
    do {
      ch = is.get();
    } while (isspace(ch));
    if (ch == ',') {
      while (true) {
        ch = is.peek();
        if (isspace(ch)) {
          is.get(); continue;
        }
        if (ch == ')') {
          is.get(); break;
        }
        break;
      }
      if (ch == ')') break;
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  shape.CopyFrom(tmp.begin(), tmp.end());
  return is;
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_SHAPE_H_
