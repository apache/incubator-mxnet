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
 * \file runtime/packed_func.h
 * \brief Type-erased function used across MXNET API.
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_RUNTIME_PACKED_FUNC_H_
#define MXNET_RUNTIME_PACKED_FUNC_H_

#include <dmlc/logging.h>
#include <mxnet/runtime/c_runtime_api.h>
#include <mxnet/runtime/object.h>
#include <mxnet/runtime/ndarray.h>
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/container_ext.h>
#include <mxnet/runtime/ndarray_handle.h>
#include <mxnet/runtime/ffi_helper.h>
#include <mxnet/runtime/data_type.h>
#include <mxnet/runtime/py_arg.h>
#include <mxnet/node/container.h>
#include <mxnet/ir/expr.h>
#include <mxnet/ndarray.h>
#include <mxnet/base.h>
#include <functional>
#include <tuple>
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <utility>
#include <type_traits>
#include <sstream>

namespace mxnet {
// forward declarations
// class Integer;
// class Expr;

namespace runtime {

/*!
 * \brief convert a string to TVM type.
 * \param s The string to be converted.
 * \return The corresponding tvm type.
 */
inline DLDataType String2DLDataType(std::string s);

// forward declarations
class MXNetArgs;
class MXNetArgValue;
class MXNetRetValue;
class MXNetArgsSetter;

/*!
 * \brief Packed function is a type-erased function.
 *  The arguments are passed by packed format.
 *
 *  This is an useful unified interface to call generated functions,
 *  It is the unified function function type of TVM.
 *  It corresponds to TVMFunctionHandle in C runtime API.
 */
class PackedFunc {
 public:
  /*!
   * \brief The internal std::function
   * \param args The arguments to the function.
   * \param rv The return value.
   *
   * \code
   *   // Example code on how to implemented FType
   *   void MyPackedFunc(MXNetArgs args, MXNetRetValue* rv) {
   *     // automatically convert arguments to desired type.
   *     int a0 = args[0];
   *     float a1 = args[1];
   *     ...
   *     // automatically assign values to rv
   *     std::string my_return_value = "x";
   *     *rv = my_return_value;
   *   }
   * \endcode
   */
  using FType = std::function<void (MXNetArgs args, MXNetRetValue* rv)>;
  /*! \brief default constructor */
  PackedFunc() {}
  /*! \brief constructor from null */
  PackedFunc(std::nullptr_t null) {}  // NOLINT(*)
  /*!
   * \brief constructing a packed function from a std::function.
   * \param body the internal container of packed function.
   */
  explicit PackedFunc(FType body) : body_(body) {}
  /*!
   * \brief Call packed function by directly passing in unpacked format.
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call packed function
   *   void CallPacked(PackedFunc f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template<typename... Args>
  inline MXNetRetValue operator()(Args&& ...args) const;
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  inline void CallPacked(MXNetArgs args, MXNetRetValue* rv) const;
  /*! \return the internal body function */
  inline FType body() const;
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const {
    return body_ == nullptr;
  }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const {
    return body_ != nullptr;
  }

 private:
  /*! \brief internal container of packed function */
  FType body_;
};

/*!
 * \brief Please refer to \ref TypedPackedFuncAnchor "TypedPackedFunc<R(Args..)>"
 */
template<typename FType>
class TypedPackedFunc;

/*!
 * \anchor TypedPackedFuncAnchor
 * \brief A PackedFunc wrapper to provide typed function signature.
 * It is backed by a PackedFunc internally.
 *
 * TypedPackedFunc enables compile time type checking.
 * TypedPackedFunc works with the runtime system:
 * - It can be passed as an argument of PackedFunc.
 * - It can be assigned to MXNetRetValue.
 * - It can be directly converted to a type-erased PackedFunc.
 *
 * Developers should prefer TypedPackedFunc over PackedFunc in C++ code
 * as it enables compile time checking.
 * We can construct a TypedPackedFunc from a lambda function
 * with the same signature.
 *
 * \code
 *  // user defined lambda function.
 *  auto addone = [](int x)->int {
 *    return x + 1;
 *  };
 *  // We can directly convert
 *  // lambda function to TypedPackedFunc
 *  TypedPackedFunc<int(int)> ftyped(addone);
 *  // invoke the function.
 *  int y = ftyped(1);
 *  // Can be directly converted to PackedFunc
 *  PackedFunc packed = ftype;
 * \endcode
 * \tparam R The return value of the function.
 * \tparam Args The argument signature of the function.
 */
template<typename R, typename ...Args>
class TypedPackedFunc<R(Args...)> {
 public:
  /*! \brief short hand for this function type */
  using TSelf = TypedPackedFunc<R(Args...)>;
  /*! \brief default constructor */
  TypedPackedFunc() {}
  /*! \brief constructor from null */
  TypedPackedFunc(std::nullptr_t null) {}  // NOLINT(*)
  /*!
   * \brief construct by wrap a PackedFunc
   *
   * Example usage:
   * \code
   * PackedFunc packed([](MXNetArgs args, MXNetRetValue *rv) {
   *   int x = args[0];
   *   *rv = x + 1;
   *  });
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(packed);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param packed The packed function
   */
  inline TypedPackedFunc(PackedFunc packed);  // NOLINT(*)
  /*!
   * \brief constructor from MXNetRetValue
   * \param value The MXNetRetValue
   */
  inline TypedPackedFunc(const MXNetRetValue& value);  // NOLINT(*)
  /*!
   * \brief constructor from MXNetArgValue
   * \param value The MXNetArgValue
   */
  inline TypedPackedFunc(const MXNetArgValue& value);  // NOLINT(*)
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(typed_lambda);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  TypedPackedFunc(const FLambda& typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
  }
  /*!
   * \brief copy assignment operator from typed lambda
   *
   * Example usage:
   * \code
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped;
   * ftyped = [](int x) { return x + 1; }
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   * \returns reference to self.
   */
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
    return *this;
  }
  /*!
   * \brief copy assignment operator from PackedFunc.
   * \param packed The packed function.
   * \returns reference to self.
   */
  TSelf& operator=(PackedFunc packed) {
    packed_ = packed;
    return *this;
  }
  /*!
   * \brief Invoke the operator.
   * \param args The arguments
   * \returns The return value.
   */
  inline R operator()(Args ...args) const;
  /*!
   * \brief convert to PackedFunc
   * \return the internal PackedFunc
   */
  operator PackedFunc() const {
    return packed();
  }
  /*!
   * \return reference the internal PackedFunc
   */
  const PackedFunc& packed() const {
    return packed_;
  }
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const {
    return packed_ == nullptr;
  }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const {
    return packed_ != nullptr;
  }

 private:
  friend class MXNetRetValue;
  /*! \brief The internal packed function */
  PackedFunc packed_;
  /*!
   * \brief Assign the packed field using a typed lambda function.
   *
   * \param flambda The lambda function.
   * \tparam FLambda The lambda function type.
   * \note We capture the lambda when possible for maximum efficiency.
   */
  template<typename FLambda>
  inline void AssignTypedLambda(FLambda flambda);
};

/*! \brief Arguments into TVM functions. */
class MXNetArgs {
 public:
  const MXNetValue* values;
  const int* type_codes;
  int num_args;
  /*!
   * \brief constructor
   * \param values The argument values
   * \param type_codes The argument type codes
   * \param num_args number of arguments.
   */
  MXNetArgs(const MXNetValue* values,
          const int* type_codes,
          int num_args)
      : values(values),
        type_codes(type_codes),
        num_args(num_args) { }
  /*! \return size of the arguments */
  inline int size() const;
  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  inline MXNetArgValue operator[](int i) const;
};

/*!
 * \brief Convert type code to its name
 * \param type_code The type code .
 * \return The name of type code.
 */
inline const char* TypeCode2Str(int type_code);

/*!
 * \brief convert a string to TVM type.
 * \param s The string to be converted.
 * \return The corresponding tvm type.
 */
// inline TVMType String2TVMType(std::string s);

// macro to check type code.
#define MXNET_CHECK_TYPE_CODE(CODE, T)                           \
  CHECK_EQ(CODE, T) << " expected "                            \
  << TypeCode2Str(T) << " but get " << TypeCode2Str(CODE)      \

/*!
 * \brief Type traits to mark if a class is tvm extension type.
 *
 * To enable extension type in C++ must be registered via marco.
 * TVM_REGISTER_EXT_TYPE(TypeName) after defining this with this traits.
 *
 * Extension class can be passed and returned via PackedFunc in all tvm runtime.
 * Internally extension class is stored as T*.
 *
 * \tparam T the typename
 */
template<typename T>
struct extension_type_info {
  static const int code = 0;
};

/*!
 * \brief Type traits for runtime type check during FFI conversion.
 * \tparam T the type to be checked.
 */
template <typename T>
struct ObjectTypeChecker {
  static bool Check(const Object* ptr) {
    using ContainerType = typename T::ContainerType;
    if (ptr == nullptr) return T::_type_is_nullable;
    return ptr->IsInstance<ContainerType>();
  }
  static std::string TypeName() {
    using ContainerType = typename T::ContainerType;
    return ContainerType::_type_key;
  }
};

/*!
 * \brief Internal base class to
 *  handle conversion to POD values.
 */
class MXNetPODValue_ {
 public:
  operator double() const {
    // Allow automatic conversion from int to float
    // This avoids errors when user pass in int from
    // the frontend while the API expects a float.
    if (type_code_ == kDLInt) {
      return static_cast<double>(value_.v_int64);
    }
    MXNET_CHECK_TYPE_CODE(type_code_, kDLFloat);
    return value_.v_float64;
  }
  operator int64_t() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator uint64_t() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator int() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kDLInt);
    CHECK_LE(value_.v_int64,
             std::numeric_limits<int>::max());
    return static_cast<int>(value_.v_int64);
  }
  operator bool() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64 != 0;
  }
  operator void*() const {
    if (type_code_ == kNull) return nullptr;
    MXNET_CHECK_TYPE_CODE(type_code_, kHandle);
    return value_.v_handle;
  }
  operator ObjectRef() const {
    if (type_code_ == kNull) {
      return ObjectRef(ObjectPtr<Object>(nullptr));
    }
    MXNET_CHECK_TYPE_CODE(type_code_, kObjectHandle);
    return ObjectRef(
        ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  }
  template<typename TObjectRef,
           typename = typename std::enable_if<
             std::is_class<TObjectRef>::value>::type>
  inline bool IsObjectRef() const;
  template <typename TObjectRef>
  inline TObjectRef AsObjectRef() const;
  int type_code() const {
    return type_code_;
  }

  /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
  template<typename T>
  T* ptr() const {
    return static_cast<T*>(value_.v_handle);
  }

 protected:
  friend class MXNetArgsSetter;
  friend class MXNetRetValue;
  MXNetPODValue_() : type_code_(kNull) {}
  MXNetPODValue_(MXNetValue value, int type_code)
      : value_(value), type_code_(type_code) {}

  /*! \brief The value */
  MXNetValue value_;
  /*! \brief the type code */
  int type_code_;
};

/*!
 * \brief A single argument value to PackedFunc.
 *  Containing both type_code and MXNetValue
 *
 *  Provides utilities to do type cast into other types.
 */
class MXNetArgValue : public MXNetPODValue_ {
 public:
  /*! \brief default constructor */
  MXNetArgValue() {}
  /*!
   * \brief constructor
   * \param value of the function
   * \param type_code The type code.
   */
  MXNetArgValue(MXNetValue value, int type_code)
      : MXNetPODValue_(value, type_code) {
  }
  // reuse converter from parent
  using MXNetPODValue_::operator double;
  using MXNetPODValue_::operator int64_t;
  using MXNetPODValue_::operator uint64_t;
  using MXNetPODValue_::operator int;
  using MXNetPODValue_::operator bool;
  using MXNetPODValue_::operator void*;
  using MXNetPODValue_::operator ObjectRef;
  using MXNetPODValue_::IsObjectRef;
  using MXNetPODValue_::AsObjectRef;

  // conversion operator.
  operator std::string() const {
    if (type_code_ == kBytes) {
      MXNetByteArray* arr = static_cast<MXNetByteArray*>(value_.v_handle);
      return std::string(arr->data, arr->size);
    } else {
      MXNET_CHECK_TYPE_CODE(type_code_, kStr);
      return std::string(value_.v_str);
    }
  }
  operator DLDataType() const {
    if (type_code_ == kStr) {
      return String2DLDataType(operator std::string());
    }
    // None type
    if (type_code_ == kNull) {
      DLDataType t;
      t.code = kHandle; t.bits = 0; t.lanes = 0;
      return t;
    }
    MXNET_CHECK_TYPE_CODE(type_code_, kMXNetType);
    return value_.v_type;
  }
  operator MXNetDataType() const {
    return MXNetDataType(operator DLDataType());
  }
  operator ::mxnet::NDArray*() const {
    if (type_code_ == kNull) {
      return nullptr;
    }
    MXNET_CHECK_TYPE_CODE(type_code_, kNDArrayHandle);
    return reinterpret_cast<::mxnet::NDArray*>(value_.v_handle);
  }
  template<typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  const MXNetValue& value() const {
    return value_;
  }
  template<typename T,
           typename = typename std::enable_if<
           std::is_class<T>::value>::type>
  inline operator T() const;
};

/*!
 * \brief Return Value container,
 *  Unlike MXNetArgValue, which only holds reference and do not delete
 *  the underlying container during destruction.
 *
 *  MXNetRetValue holds value and will manage the underlying containers
 *  when it stores a complicated data type.
 */
class MXNetRetValue : public MXNetPODValue_ {
 public:
  /*! \brief default constructor */
  MXNetRetValue() {}
  /*!
   * \brief move constructor from anoter return value.
   * \param other The other return value.
   */
  MXNetRetValue(MXNetRetValue&& other)
      : MXNetPODValue_(other.value_, other.type_code_) {
    other.value_.v_handle = nullptr;
    other.type_code_ = kNull;
  }
  /*! \brief destructor */
  ~MXNetRetValue() {
    this->Clear();
  }
  // reuse converter from parent
  using MXNetPODValue_::operator double;
  using MXNetPODValue_::operator int64_t;
  using MXNetPODValue_::operator uint64_t;
  using MXNetPODValue_::operator int;
  using MXNetPODValue_::operator bool;
  using MXNetPODValue_::operator void*;
  using MXNetPODValue_::operator ObjectRef;
  using MXNetPODValue_::IsObjectRef;
  using MXNetPODValue_::AsObjectRef;

  MXNetRetValue(const MXNetRetValue& other) : MXNetPODValue_() {
    this->Assign(other);
  }
  // conversion operators
  operator std::string() const {
    if (type_code_ == kBytes) {
      return *ptr<std::string>();
    }
    MXNET_CHECK_TYPE_CODE(type_code_, kStr);
    return *ptr<std::string>();
  }
  operator DLDataType() const {
    if (type_code_ == kStr) {
      return String2DLDataType(operator std::string());
    }
    MXNET_CHECK_TYPE_CODE(type_code_, kMXNetType);
    return value_.v_type;
  }
  operator MXNetDataType() const {
    return MXNetDataType(operator DLDataType());
  }
  template<typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  // Assign operators
  MXNetRetValue& operator=(MXNetRetValue&& other) {
    this->Clear();
    value_ = other.value_;
    type_code_ = other.type_code_;
    other.type_code_ = kNull;
    return *this;
  }
  MXNetRetValue& operator=(double value) {
    this->SwitchToPOD(kDLFloat);
    value_.v_float64 = value;
    return *this;
  }
  MXNetRetValue& operator=(std::nullptr_t value) {
    this->SwitchToPOD(kNull);
    value_.v_handle = value;
    return *this;
  }
  MXNetRetValue& operator=(void* value) {
    this->SwitchToPOD(kHandle);
    value_.v_handle = value;
    return *this;
  }
  MXNetRetValue& operator=(int64_t value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  MXNetRetValue& operator=(int value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  MXNetRetValue& operator=(bool value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  MXNetRetValue& operator=(std::string value) {
    this->SwitchToClass(kStr, value);
    return *this;
  }
  MXNetRetValue& operator=(DLDataType t) {
    this->SwitchToPOD(kMXNetType);
    value_.v_type = t;
    return *this;
  }
  MXNetRetValue& operator=(const MXNetDataType& other) {
    return operator=(other.operator DLDataType());
  }
  MXNetRetValue& operator=(MXNetByteArray value) {
    this->SwitchToClass(kBytes, std::string(value.data, value.size));
    return *this;
  }
  MXNetRetValue& operator=(ObjectRef other) {
    if (other.as<NDArrayHandleObj>()) {
      return operator=(Downcast<NDArrayHandle, ObjectRef>(other));
    }
    return operator=(std::move(other.data_));
  }
  template<typename T>
  MXNetRetValue& operator=(ObjectPtr<T> other) {
    SwitchToObject(kObjectHandle, std::move(other));
    return *this;
  }
  template<typename FType>
  MXNetRetValue& operator=(const TypedPackedFunc<FType>& f) {
    return operator=(f.packed());
  }
  MXNetRetValue& operator=(const MXNetRetValue& other) {  // NOLINT(*0
    this->Assign(other);
    return *this;
  }
  MXNetRetValue& operator=(const MXNetArgValue& other) {
    this->Assign(other);
    return *this;
  }
  MXNetRetValue& operator=(NDArray* value) {
    this->SwitchToPOD(kNDArrayHandle);
    value_.v_handle = reinterpret_cast<void*>(value);
    return *this;
  }
  MXNetRetValue& operator=(NDArrayHandle value) {
    this->SwitchToPOD(kNDArrayHandle);
    NDArray* arr = new NDArray(value->value);
    value_.v_handle = reinterpret_cast<void*>(arr);
    return *this;
  }
  MXNetRetValue& operator=(const PythonArg& value) {
    this->SwitchToPOD(kPyArg);
    value_.v_int64 = value.offset();
    return *this;
  }
  template<typename T,
           typename = typename std::enable_if<
             extension_type_info<T>::code != 0>::type>
  MXNetRetValue& operator=(const T& other) {
    this->SwitchToClass<T>(
        extension_type_info<T>::code, other);
    return *this;
  }
  /*!
   * \brief Move the value back to front-end via C API.
   *  This marks the current container as null.
   *  The managed resources is moved to front-end and
   *  the front end should take charge in managing them.
   *
   * \param ret_value The return value.
   * \param ret_type_code The return type code.
   */
  void MoveToCHost(MXNetValue* ret_value,
                   int* ret_type_code) {
    // cannot move str; need specially handle.
    CHECK(type_code_ != kStr && type_code_ != kBytes);
    *ret_value = value_;
    *ret_type_code = type_code_;
    type_code_ = kNull;
  }
  /*! \return The value field, if the data is POD */
  const MXNetValue& value() const {
    CHECK(type_code_ != kObjectHandle &&
          type_code_ != kStr) << "MXNetRetValue.value can only be used for POD data";
    return value_;
  }
  // ObjectRef related extenstions: in tvm/packed_func_ext.h
  template<typename T,
           typename = typename std::enable_if<
             std::is_class<T>::value>::type>
  inline operator T() const;

 private:
  template<typename T>
  void Assign(const T& other) {
    switch (other.type_code()) {
      case kStr: {
        SwitchToClass<std::string>(kStr, other);
        break;
      }
      case kBytes: {
        SwitchToClass<std::string>(kBytes, other);
        break;
      }
      case kObjectHandle: {
        *this = other.operator ObjectRef();
        break;
      }
      default: {
        if (other.type_code() < kExtBegin) {
          SwitchToPOD(other.type_code());
          value_ = other.value_;
        } else {
          LOG(FATAL) << "Does not support ext type";
        }
        break;
      }
    }
  }
  // get the internal container.
  void SwitchToPOD(int type_code) {
    if (type_code_ != type_code) {
      this->Clear();
      type_code_ = type_code;
    }
  }
  template<typename T>
  void SwitchToClass(int type_code, T v) {
    if (type_code_ != type_code) {
      this->Clear();
      type_code_ = type_code;
      value_.v_handle = new T(v);
    } else {
      *static_cast<T*>(value_.v_handle) = v;
    }
  }
  void SwitchToObject(int type_code, ObjectPtr<Object> other) {
    if (other.data_ != nullptr) {
      this->Clear();
      type_code_ = type_code;
      // move the handle out
      value_.v_handle = other.data_;
      other.data_ = nullptr;
    } else {
      SwitchToPOD(kNull);
    }
  }
  void Clear() {
    if (type_code_ == kNull) return;
    switch (type_code_) {
      case kStr: delete ptr<std::string>(); break;
      case kObjectHandle: {
        static_cast<Object*>(value_.v_handle)->DecRef();
        break;
      }
    }
    if (type_code_ > kExtBegin) {
      LOG(FATAL) << "Does not support ext type";
    }
    type_code_ = kNull;
  }
};

inline DLDataType String2DLDataType(std::string s) {
  DLDataType t;
  // handle None type
  if (s.length() == 0) {
    t.bits = 0; t.lanes = 0; t.code = kHandle;
    return t;
  }
  t.bits = 32; t.lanes = 1;
  const char* scan = nullptr;
  if (s.substr(0, 3) == "int") {
    t.code = kDLInt;  scan = s.c_str() + 3;
  } else if (s.substr(0, 4) == "uint") {
    t.code = kDLUInt; scan = s.c_str() + 4;
  } else if (s.substr(0, 5) == "float") {
    t.code = kDLFloat; scan = s.c_str() + 5;
  } else if (s.substr(0, 6) == "handle") {
    t.code = kHandle;
    t.bits = 64;  // handle uses 64 bit by default.
    scan = s.c_str() + 6;
  } else if (s == "bool") {
    t.code = kDLUInt;
    t.bits = 1;
    t.lanes = 1;
    return t;
  } else if (s.substr(0, 6) == "custom") {
    LOG(FATAL) << "custom MXNetDataType is not supported";
    // t.code = ParseCustomDatatype(s, &scan);
  } else {
    scan = s.c_str();
    LOG(FATAL) << "unknown type " << s;
  }
  char* xdelim;  // emulate sscanf("%ux%u", bits, lanes)
  uint8_t bits = static_cast<uint8_t>(strtoul(scan, &xdelim, 10));
  if (bits != 0) t.bits = bits;
  char* endpt = xdelim;
  if (*xdelim == 'x') {
    t.lanes = static_cast<uint16_t>(strtoul(xdelim + 1, &endpt, 10));
  }
  CHECK(endpt == s.c_str() + s.length()) << "unknown type " << s;
  return t;
}

// implementation details
inline const char* TypeCode2Str(int type_code) {
  switch (type_code) {
    case kDLInt: return "int";
    case kDLUInt: return "uint";
    case kDLFloat: return "float";
    case kStr: return "str";
    case kBytes: return "bytes";
    case kHandle: return "handle";
    case kNull: return "NULL";
    case kObjectHandle: return "ObjectCell";
    case kNDArrayHandle: return "NDArray";
    default: LOG(FATAL) << "unknown type_code="
                        << static_cast<int>(type_code); return "";
  }
}

inline int String2MXNetTypeWithBool(const std::string& s) {
  if (s == "float32") {
    return mshadow::kFloat32;
  } else if (s == "float64") {
    return mshadow::kFloat64;
  } else if (s == "float16") {
    return mshadow::kFloat16;
  } else if (s == "uint8") {
    return mshadow::kUint8;
  } else if (s == "int8") {
    return mshadow::kInt8;
  } else if (s == "uint16") {
    return mshadow::kUint16;
  } else if (s == "int16") {
    return mshadow::kInt16;
  } else if (s == "int32") {
    return mshadow::kInt32;
  } else if (s == "int64") {
    return mshadow::kInt64;
  } else if (s == "bool") {
    return mshadow::kBool;
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

inline int String2MXNetType(const std::string& s) {
  if (s == "float32") {
    return mshadow::kFloat32;
  } else if (s == "float64") {
    return mshadow::kFloat64;
  } else if (s == "float16") {
    return mshadow::kFloat16;
  } else if (s == "uint8") {
    return mshadow::kUint8;
  } else if (s == "int8") {
    return mshadow::kInt8;
  } else if (s == "uint16") {
    return mshadow::kUint16;
  } else if (s == "int16") {
    return mshadow::kInt16;
  } else if (s == "int32") {
    return mshadow::kInt32;
  } else if (s == "int64") {
    return mshadow::kInt64;
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

inline std::ostream& operator<<(std::ostream& os, DLDataType t) {  // NOLINT(*)
  if (t.bits == 1 && t.lanes == 1 && t.code == kDLUInt) {
    os << "bool"; return os;
  }
  if (t.code < kCustomBegin) {
    os << TypeCode2Str(t.code);
  } else {
    LOG(FATAL) << "custom MXNetDataType is not supported";
    // os << "custom[" << GetCustomTypeName(t.code) << "]";
  }
  if (t.code == kHandle) return os;
  os << static_cast<int>(t.bits);
  if (t.lanes != 1) {
    os << 'x' << static_cast<int>(t.lanes);
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const MXNetDataType& dtype) { // NOLINT(*)
  return os << dtype.operator DLDataType();
}

inline MXNetArgValue MXNetArgs::operator[](int i) const {
  CHECK_LT(i, num_args)
      << "not enough argument passed, "
      << num_args << " passed"
      << " but request arg[" << i << "].";
  return MXNetArgValue(values[i], type_codes[i]);
}

inline int MXNetArgs::size() const {
  return num_args;
}

inline void PackedFunc::CallPacked(MXNetArgs args, MXNetRetValue* rv) const {
  body_(args, rv);
}

inline PackedFunc::FType PackedFunc::body() const {
  return body_;
}

// internal namespace
namespace detail {

template<bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template<typename T, typename ...Args>
  static void run(const F& f, T&& value, Args&&... args) {  // NOLINT(*)
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, (I+1), F>
        ::run(f, std::forward<Args>(args)...);
  }
};

template<std::size_t I, typename F>
struct for_each_dispatcher<true, I, F>  {
  static void run(const F& f) {}  // NOLINT(*)
};

template<typename F, typename ...Args>
inline void for_each(const F& f, Args&&... args) {  // NOLINT(*)
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>
      ::run(f, std::forward<Args>(args)...);
}
}  // namespace detail

/* \brief argument settter to PackedFunc */
class MXNetArgsSetter {
 public:
  MXNetArgsSetter(MXNetValue* values, int* type_codes)
      : values_(values), type_codes_(type_codes) {}
  // setters for POD types
  template<typename T,
           typename = typename std::enable_if<
             std::is_integral<T>::value>::type>
  void operator()(size_t i, T value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, uint64_t value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    CHECK_LE(value,
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDLFloat;
  }
  void operator()(size_t i, std::nullptr_t value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kNull;
  }
  void operator()(size_t i, const MXNetArgValue& value) const {
    values_[i] = value.value_;
    type_codes_[i] = value.type_code_;
  }
  void operator()(size_t i, void* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kHandle;
  }
  void operator()(size_t i, const char* value) const {
    values_[i].v_str = value;
    type_codes_[i] = kStr;
  }
  // setters for container type
  // They must be reference(instead of const ref)
  // to make sure they are alive in the tuple(instead of getting converted)
  void operator()(size_t i, const std::string& value) const {  // NOLINT(*)
    values_[i].v_str = value.c_str();
    type_codes_[i] = kStr;
  }
  void operator()(size_t i, DLDataType value) const {
    values_[i].v_type = value;
    type_codes_[i] = kMXNetType;
  }
  void operator()(size_t i, MXNetDataType dtype) const {
    operator()(i, dtype.operator DLDataType());
  }
  void operator()(size_t i, const MXNetByteArray& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<MXNetByteArray*>(&value);
    type_codes_[i] = kBytes;
  }
  template<typename FType>
  void operator()(size_t i, const TypedPackedFunc<FType>& value) const {  // NOLINT(*)
    operator()(i, value.packed());
  }
  void operator()(size_t i, const ObjectRef& value) const {  // NOLINT(*)
    if (value.defined()) {
      values_[i].v_handle = value.data_.data_;
      type_codes_[i] = kObjectHandle;
    } else {
      type_codes_[i] = kNull;
    }
  }
  void operator()(size_t i, const MXNetRetValue& value) const {  // NOLINT(*)
    if (value.type_code() == kStr) {
      values_[i].v_str = value.ptr<std::string>()->c_str();
      type_codes_[i] = kStr;
    } else {
      CHECK_NE(value.type_code(), kBytes) << "not handled.";
      values_[i] = value.value_;
      type_codes_[i] = value.type_code();
    }
  }

 private:
  /*! \brief The values fields */
  MXNetValue* values_;
  /*! \brief The type code fields */
  int* type_codes_;
};

template<typename... Args>
inline MXNetRetValue PackedFunc::operator()(Args&& ...args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  MXNetValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(MXNetArgsSetter(values, type_codes),
                   std::forward<Args>(args)...);
  MXNetRetValue rv;
  body_(MXNetArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

namespace detail {
template<typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template<typename ...Args>
  static void run(const F& f,
                  const MXNetArgs& args_pack,
                  MXNetRetValue* rv,
                  Args&&... unpacked_args) {
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>
        ::run(f, args_pack, rv,
              std::forward<Args>(unpacked_args)...,
              args_pack[index]);
  }
};

template<typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const MXNetArgs& args_pack,
                  MXNetRetValue* rv,
                  Args&&... unpacked_args) {
    *rv = R(f(std::forward<Args>(unpacked_args)...));
  }
};

template<int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const MXNetArgs& args_pack,
                  MXNetRetValue* rv,
                  Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template<typename R, int nargs, typename F>
inline void unpack_call(const F& f, const MXNetArgs& args, MXNetRetValue* rv) {
  unpack_call_dispatcher<R, nargs, 0, F>::run(f, args, rv);
}

template<typename R, typename ...Args>
inline R call_packed(const PackedFunc& pf, Args&& ...args) {
  return R(pf(std::forward<Args>(args)...));
}

template<typename R>
struct typed_packed_call_dispatcher {
  template<typename ...Args>
  static inline R run(const PackedFunc& pf, Args&& ...args) {
    return pf(std::forward<Args>(args)...);
  }
};

template<>
struct typed_packed_call_dispatcher<void> {
  template<typename ...Args>
  static inline void run(const PackedFunc& pf, Args&& ...args) {
    pf(std::forward<Args>(args)...);
  }
};
}  // namespace detail

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(PackedFunc packed)
  : packed_(packed) {}

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(const MXNetRetValue& value)
    : packed_(value.operator PackedFunc()) {}

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(const MXNetArgValue& value)
    : packed_(value.operator PackedFunc()) {}

template<typename R, typename ...Args>
template<typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda) {
  packed_ = PackedFunc([flambda](const MXNetArgs& args, MXNetRetValue* rv) {
      detail::unpack_call<R, sizeof...(Args)>(flambda, args, rv);
    });
}

template<typename R, typename ...Args>
inline R TypedPackedFunc<R(Args...)>::operator()(Args... args) const {
  return detail::typed_packed_call_dispatcher<R>
      ::run(packed_, std::forward<Args>(args)...);
}

// extension and node type handling
namespace detail {
template<typename T, typename TSrc, bool is_ext, bool is_nd>
struct MXNetValueCast {
  static T Apply(const TSrc* self) {
    static_assert(!is_ext && !is_nd, "The default case accepts only non-extensions");
    return self->template AsObjectRef<T>();
  }
};

}  // namespace detail

/*!
 * \brief Type trait to specify special value conversion rules from
 *        MXNetArgValue and MXNetRetValue.
 *
 *  The trait can be specialized to add type specific conversion logic
 *  from the TVMArgvalue and TVMRetValue.
 *
 * \tparam TObjectRef the specific ObjectRefType.
 */
template <typename TObjectRef>
struct PackedFuncValueConverter {
  /*!
   * \brief Convert a TObjectRef from an argument value.
   * \param val The argument value.
   * \return the converted result.
   */
  static TObjectRef From(const MXNetArgValue& val) { return val.AsObjectRef<TObjectRef>(); }
  /*!
   * \brief Convert a TObjectRef from a return value.
   * \param val The argument value.
   * \return the converted result.
   */
  static TObjectRef From(const MXNetRetValue& val) { return val.AsObjectRef<TObjectRef>(); }
};

template <>
struct PackedFuncValueConverter<::mxnet::runtime::String> {
  static String From(const MXNetArgValue& val) {
    if (val.IsObjectRef<mxnet::runtime::String>()) {
      return val.AsObjectRef<mxnet::runtime::String>();
    } else {
      return mxnet::runtime::String(val.operator std::string());
    }
  }

  static String From(const MXNetRetValue& val) {
    if (val.IsObjectRef<mxnet::runtime::String>()) {
      return val.AsObjectRef<mxnet::runtime::String>();
    } else {
      return mxnet::runtime::String(val.operator std::string());
    }
  }
};

template <typename TObjectRef>
inline TObjectRef MXNetPODValue_::AsObjectRef() const {
  static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                "Conversion only works for ObjectRef");
  using ContainerType = typename TObjectRef::ContainerType;

  if (type_code_ == kNull) {
    CHECK(TObjectRef::_type_is_nullable)
        << "Expect a not null value of " << ContainerType::_type_key;
    return TObjectRef(ObjectPtr<Object>(nullptr));
  }
  if (type_code_ == kObjectHandle) {
    // normal object type check.
    Object* ptr = static_cast<Object*>(value_.v_handle);
    CHECK(ObjectTypeChecker<TObjectRef>::Check(ptr))
        << "Expect " << ObjectTypeChecker<TObjectRef>::TypeName() << " but get "
        << ptr->GetTypeKey();
    return TObjectRef(GetObjectPtr<Object>(ptr));
  } else {
    MXNET_CHECK_TYPE_CODE(type_code_, kObjectHandle);
    return TObjectRef(ObjectPtr<Object>(nullptr));
  }
}

template <typename T, typename>
inline MXNetArgValue::operator T() const {
  return PackedFuncValueConverter<T>::From(*this);
}

template <typename TObjectRef, typename>
inline bool MXNetPODValue_::IsObjectRef() const {
  using ContainerType = typename TObjectRef::ContainerType;
  return  type_code_ == kObjectHandle &&
          ObjectTypeChecker<TObjectRef>::Check(static_cast<Object*>(value_.v_handle));
}

inline bool String::CanConvertFrom(const MXNetArgValue& val) {
  return val.type_code() == kStr || val.IsObjectRef<mxnet::runtime::String>();
}

}  // namespace runtime
}  // namespace mxnet
#endif  // MXNET_RUNTIME_PACKED_FUNC_H_
