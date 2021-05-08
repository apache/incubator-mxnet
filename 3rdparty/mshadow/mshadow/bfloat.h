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
 *  Copyright (c) 2019 by Contributors
 * \file bfloat.h
 * \brief definition of bfloat type.
 *
 * \author Zhennan Qin
 */
#ifndef MSHADOW_BFLOAT_H_
#define MSHADOW_BFLOAT_H_
#include "./base.h"

/*! \brief namespace for mshadow */
namespace mshadow {
/* \brief name space for host/device portable bfloats */
namespace bfloat {

#define MSHADOW_BF16_OPERATOR_TYPE(RTYPE, ITYPE, OP)                      \
  MSHADOW_XINLINE RTYPE operator OP (ITYPE a, bf16_t b) {                 \
    return RTYPE(a OP float(b));  /* NOLINT(*) */                         \
  }                                                                       \
  MSHADOW_XINLINE RTYPE operator OP (bf16_t a, ITYPE b) {                 \
    return RTYPE(float(a) OP b);  /* NOLINT(*) */                         \
  }

#define MSHADOW_BF16_OPERATOR(RTYPE, OP)                                  \
  MSHADOW_XINLINE RTYPE operator OP (bf16_t a, bf16_t b) {                \
    return RTYPE(static_cast<float>(a) OP float(b));  /* NOLINT(*) */     \
  }                                                                       \
  MSHADOW_BF16_OPERATOR_TYPE(float, float, OP)                            \
  MSHADOW_BF16_OPERATOR_TYPE(double, double, OP)                          \
  MSHADOW_BF16_OPERATOR_TYPE(float, int8_t, OP)                           \
  MSHADOW_BF16_OPERATOR_TYPE(float, uint8_t, OP)                          \
  MSHADOW_BF16_OPERATOR_TYPE(float, int16_t, OP)                           \
  MSHADOW_BF16_OPERATOR_TYPE(float, uint16_t, OP)                          \
  MSHADOW_BF16_OPERATOR_TYPE(float, int32_t, OP)                          \
  MSHADOW_BF16_OPERATOR_TYPE(float, uint32_t, OP)                         \
  MSHADOW_BF16_OPERATOR_TYPE(float, int64_t, OP)                          \
  MSHADOW_BF16_OPERATOR_TYPE(float, uint64_t, OP)

#define MSHADOW_BF16_ASSIGNOP(AOP, OP)                                    \
  template<typename T>                                                    \
  MSHADOW_XINLINE bf16_t operator AOP (const T& a) {                      \
    return *this = bf16_t(float(*this) OP float(a));  /* NOLINT(*)*/      \
  }                                                                       \
  template<typename T>                                                    \
  MSHADOW_XINLINE bf16_t operator AOP (const volatile T& a) volatile {    \
    return *this = bf16_t(float(*this) OP float(a));  /* NOLINT(*)*/      \
  }

#define MSHADOW_BF16_CONVERSIONOP(T)                                      \
  MSHADOW_XINLINE operator T() const {                                    \
    return T(BF16ToFloat(bf16_));  /* NOLINT(*)*/                            \
  }                                                                       \
  MSHADOW_XINLINE operator T() const volatile {                           \
    return T(BF16ToFloat(bf16_));  /* NOLINT(*)*/                            \
  }

class MSHADOW_ALIGNED(2) bf16_t {
 public:
  uint16_t bf16_;

static MSHADOW_XINLINE bf16_t Binary(uint16_t value) {
  bf16_t res;
  res.bf16_ = value;
  return res;
  }

  MSHADOW_XINLINE bf16_t() {}

  MSHADOW_XINLINE bf16_t(const float& value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const double& value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const int8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const uint8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const int16_t &value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const uint16_t &value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const int32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const uint32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const int64_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit bf16_t(const uint64_t& value) { constructor(value); }

  MSHADOW_BF16_CONVERSIONOP(float)

  MSHADOW_BF16_ASSIGNOP(+=, +)
  MSHADOW_BF16_ASSIGNOP(-=, -)
  MSHADOW_BF16_ASSIGNOP(*=, *)
  MSHADOW_BF16_ASSIGNOP(/=, /)

  MSHADOW_XINLINE bf16_t operator+() {
    return *this;
  }

  MSHADOW_XINLINE bf16_t operator-() {
    return bf16_t(-float(*this));  // NOLINT(*)
  }

  MSHADOW_XINLINE bf16_t operator=(const bf16_t& a) {
    bf16_ = a.bf16_;
    return a;
  }

  template<typename T>
  MSHADOW_XINLINE bf16_t operator=(const T& a) {
    return *this = bf16_t(a);  /* NOLINT(*)*/
  }

  MSHADOW_XINLINE bf16_t operator=(const bf16_t& a) volatile {
    bf16_ = a.bf16_;
    return a;
  }

  template<typename T>
  MSHADOW_XINLINE bf16_t operator=(const T& a) volatile {
    return *this = bf16_t(a);  /* NOLINT(*)*/
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  MSHADOW_XINLINE uint16_t FloatToBF16(const float& value) const {
    return reinterpret_cast<const uint16_t*>(&value)[1];
  }

  // Same as above routine, except for addition of volatile keyword
  MSHADOW_XINLINE uint16_t FloatToBF16(const volatile float& value) const volatile {  // NOLINT (*)
    return reinterpret_cast<const volatile uint16_t*>(&value)[1];
  }

  MSHADOW_XINLINE float BF16ToFloat(const uint16_t& value) const {
    float ret = 0.f;
    reinterpret_cast<uint16_t*>(&ret)[1] = value;
    return ret;
  }

  MSHADOW_XINLINE float BF16ToFloat(const volatile uint16_t& value) const volatile {  // NOLINT(*)
    float ret = 0.f;
    reinterpret_cast<uint16_t*>(&ret)[1] = value;
    return ret;
  }

  template<typename T>
  MSHADOW_XINLINE void constructor(const T& value) {
    bf16_ = FloatToBF16(float(value));  // NOLINT(*)
  }
};

/*! \brief overloaded + operator for bf16_t */
MSHADOW_BF16_OPERATOR(bf16_t, +)
/*! \brief overloaded - operator for bf16_t */
MSHADOW_BF16_OPERATOR(bf16_t, -)
/*! \brief overloaded * operator for bf16_t */
MSHADOW_BF16_OPERATOR(bf16_t, *)
/*! \brief overloaded / operator for bf16_t */
MSHADOW_BF16_OPERATOR(bf16_t, /)
/*! \brief overloaded > operator for bf16_t */
MSHADOW_BF16_OPERATOR(bool, >)
/*! \brief overloaded < operator for bf16_t */
MSHADOW_BF16_OPERATOR(bool, <)
/*! \brief overloaded >= operator for bf16_t */
MSHADOW_BF16_OPERATOR(bool, >=)
/*! \brief overloaded <= operator for bf16_t */
MSHADOW_BF16_OPERATOR(bool, <=)

#define MSHADOW_BF16_MIN mshadow::bfloat::bf16_t::Binary(0xFF7F);
#define MSHADOW_BF16_MAX mshadow::bfloat::bf16_t::Binary(0x7F7F);
}  // namespace bfloat
}  // namespace mshadow
#endif  // MSHADOW_BFLOAT_H_