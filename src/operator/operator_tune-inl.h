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
#ifndef MXNET_OPERATOR_OPERATOR_TUNE_INL_H_
#define MXNET_OPERATOR_OPERATOR_TUNE_INL_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mshadow/base.h>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <list>
#include <random>
#include <unordered_set>
#include "./mxnet_op.h"
#include "./operator_tune.h"

#if (__GNUC__ >= 4 || (__GNUC__ >= 3 && __GNUC_MINOR__ >= 4)) && !defined(__mips__)
#  define HAS_CXA_DEMANGLE 1
#else
#  define HAS_CXA_DEMANGLE 0
#endif

#if HAS_CXA_DEMANGLE
#include <cxxabi.h>
#endif

namespace mxnet {
namespace op {

#ifndef MXNET_NO_INLINE
#ifdef _MSC_VER
#define MXNET_NO_INLINE __declspec(noinline)
#else
#define MXNET_NO_INLINE __attribute__((noinline))
#endif
#endif  // MXNET_NO_INLINE

#define OUTSIDE_COUNT_SHIFT  3

namespace tune {

/*!
 * \brief Convert TuningMode value to a string representation
 * \param tm  Scalar TuningMode value
 * \return Character pointer to a string representing the TuningMode value
 */
inline const char *TuningModeToString(const TuningMode tm) {
  switch (tm) {
    case kAuto:
      return "Auto";
    case kNeverOMP:
      return "NeverOMP";
    case kAlwaysOMP:
      return "AlwaysOMP";
    default:
      CHECK(false) << "Unknown TuningMode type: " << static_cast<int>(tm);
      return "<unknown>";
  }
}
}  // namespace tune

/*!
 * \brief Engine to tune kernel operations
 * \tparam DType Data type to be used when tuning the kernel operations
 * \remarks The basic concept here is that we time how long a trivial loop takes with and without
 * OMP, subtracting the non-OMP run from the OMP run, which gives us the time
 * that the OMP overhead takes.  Times were found to be relatively invariant with
 * regard ot the number of threads/cores on a given machine.
 * Secondly, supplied operators are run and timed (for each data type) in order to determine
 * their individual time cost.
 *
 * Knowing the following items, we can determine how long the OMP and non-OMP run
 * is expected to take:
 *  1) OMP overhead time
 *  2) Number of iterations required
 *  3) Number of threads to be used if we choose the OMP method
 *  4) The data type
 *
 * Therefore, at Kernel::Launch() time, we can estimate whether it is faster to use OMP or not
 * for the given kernel operator.
 *
 * Results and efficiency of the tuning is tested in the gtest OMP_TUNING test suite
 */
template<typename DType>
class OperatorTune : public OperatorTuneByType<DType> {
 public:
  using Tick = OperatorTuneBase::Tick;
  using duration_t = OperatorTuneBase::duration_t;
  using OperatorTuneByType<DType>::tuning_mode_;

  /*!
   * \brief Constructor
   */
  OperatorTune() {
    TuneAll();
  }

  /*!
   * \brief Initialize the OperatorTune object
   * \return Whether the OperatorTune object was successfully initialized
   */
  static bool Initialize() {
    if (!initialized_) {
      initialized_ = true;
      // Generate some random data for calling the operator kernels
      data_set_ =
        std::unique_ptr<DType[]>(reinterpret_cast<DType*>(new char[0x100 * sizeof(DType)]));
      std::random_device rd;
      std::mt19937 gen(rd());
      if (!std::is_integral<DType>::value) {
        std::uniform_real_distribution<> dis(-1, 1);
        for (int n = 0; n < 0x100; ++n) {
          const auto val = static_cast<DType>(dis(gen));
          // If too close to zero, try again
          if (std::fabs(static_cast<double>(val)) < 1e-5) {
            --n;
            continue;
          }
          data_set_[n] = val;
        }
      } else {
        std::uniform_int_distribution<> dis(-128, 127);
        for (int n = 0; n < 0x100; ++n) {
          const auto val = static_cast<DType>(dis(gen));
          // If zero, try again
          if (!val) {
            --n;
            continue;
          }
          data_set_[n] = val;
        }
      }
      // Use this environment variable to generate new tuning statistics
      // In order to avoid printing too many copies, only the float32 object prints
      output_tuning_data_ = mshadow::DataType<DType>::kFlag == mshadow::kFloat32
                            && dmlc::GetEnv("MXNET_OUTPUT_TUNING_DATA", false);
      // If outputting tuning data, then also output verbose logging info
      OperatorTuneBase::verbose_tuning_info_ = dmlc::GetEnv("MXNET_VERBOSE_TUNING_INFO", false);

      OperatorTuneBase::tuning_weight_scale_ = dmlc::GetEnv("MXNET_TUNING_WEIGHT_SCALE", 0.0);

      // This isn't actually supposed to be multithreaded init, but just to be sure the change is
      // seen everywhere, using atomic bool.
      if (!OperatorTuneBase::calculated_.load()) {
        // Not especially concerned with a race condition, since this hsould
        // run when only one thread is active (static init), just don't cache this variable
        OperatorTuneBase::calculated_.store(true);
        std::string config = dmlc::GetEnv("MXNET_USE_OPERATOR_TUNING", std::string());
        StringUtil::trim(&config);
        // disabled
        if (!config.empty() && ::isdigit(config[0]) && std::atoi(config.c_str()) == 0) {
          OperatorTuneBase::omp_overhead_ns_ = INT_MAX;
        } else {
          OperatorTuneBase::omp_overhead_ns_ = GetOMPLoopOverhead();
        }
        ParseEnablerConfig(config);
      }

      if (OperatorTuneBase::verbose_tuning_info_) {
        LOG(INFO) << "OMP overhead: " << OperatorTuneBase::omp_overhead_ns_ << " nanoseconds";
      }
    }
    return true;
  }

  /*!
   * \brief Schedule a tuning run
   * \tparam OP Operator to tune
   * \param tune_func Function to call which tunes the operator
   * \return true if the tune operation was scheduled
   */
  template<typename OP>
  static bool ScheduleTune(void (*tune_func)()) {
#ifdef MXNET_USE_OPERATOR_TUNING
    if (tune_func) {
      GetTuningList()->push_back(tune_func);
      operator_names_.insert(demangle(typeid(OP).name()));
      return true;
    }
    return false;
#else
    return true;
#endif
  }

  /*!
   * \brief Is the template parameter type a tuned kernel?
   * \tparam OP kernel operator type
   * \return true if the operator/kernel is tuned
   */
  template<typename OP>
  static bool IsTuned() {
    return operator_names_.find(demangle(typeid(OP).name())) != operator_names_.end();
  }

  /*!\
   * \brief Tune all registered kernel operators that haven't already been tuned
   */
  static bool TuneAll() {
    Initialize();
    std::list<void (*)()> *tl = GetTuningList();
    const size_t size_save = tl->size();  // For checking if anything asynchronous is
    // adding or removing items, which is forbidden
    if (output_tuning_data_ && !tl->empty()) {
      // Only emit this once, use the most common case, 'float32'
      if (mshadow::DataType<DType>::kFlag == mshadow::kFloat32) {
        std::cout << "OperatorTuneBase::duration_t "
                  << "OperatorTuneBase::omp_overhead_ns_ = " << OperatorTuneBase::omp_overhead_ns_
                  << ";" << std::endl << std::flush;
      }
    }
    const Tick start = std::chrono::high_resolution_clock::now();
    for (auto i : *tl) {
      (*i)();
    }
    if (OperatorTuneBase::verbose_tuning_info_) {
      const duration_t duration = OperatorTune::GetDurationInNanoseconds(start);
      LOG(INFO) << "Op Tuning  for " << type_name<DType>()
                << " took " << (duration / 1000000) << " ms";
    }
    CHECK_EQ(size_save, tl->size()) << "Tuning list size should not have changed while tuning";
    tl->clear();
    return true;
  }

  /*!
   * \brief Return set of operator names that were registered to be tuned. Does not imply
   *        that the operator has been tuned.
   * \return Set of operator/kernel names that were registered for tuning
   */
  static const std::unordered_set<std::string>& TunedOperatorNames() {
    return operator_names_;
  }

 protected:
  /*!
   * \brief Get the list of tuning function calls for the operators
   * \return Pointer to list of tuning function calls
   */
  static std::list<void (*)()> *GetTuningList();

  /*!
   * \brief Demangle typeid::name() in order to generate source macros
   * \param name C++ Mangled name
   * \return Demangled name as string
   */
  static inline std::string demangle(const char *name) {
#if HAS_CXA_DEMANGLE
    int status = -4;  // some arbitrary value to eliminate the compiler warning
    std::unique_ptr<char, void (*)(void *)> res{
      abi::__cxa_demangle(name, nullptr, nullptr, &status),
      &std::free
    };
    return status ? name : res.get();
#else
    return name;
#endif
  }

  /*!
   * \brief Type name as string
   * \tparam T Type
   * \return std::string representing the human-readable demangled type name
   */
  template<typename T> static inline std::string type_name() {
    return demangle(typeid(T).name());
  }

  /*! \brief Measure OMP overhead for a trivial OMP loop using all cores
   * \param omp_thread_count - Number of OMP threads to use in the timing test
   * \returns Duration in nanoseconds for the OMP overhead (time to initiate and close the
   *          OMP session)
   */
  static duration_t GetOMPLoopOverhead(const size_t omp_thread_count) {
    CHECK_GT(omp_thread_count, 1);  // Don't try to use OMP for one thread
    int wl_count = OperatorTuneBase::WORKLOAD_COUNT;

    Tick start = std::chrono::high_resolution_clock::now();
    // Use two loops in order to simulate OMP outside timing
    for (size_t i = 0; i < OUTSIDE_COUNT; ++i) {
      for (int x = 0; x < wl_count; ++x) {
        // trivial operation
        volatile_int_ += x;
      }
    }
    const OperatorTuneBase::duration_t no_omp_duration =
      OperatorTuneBase::GetDurationInNanoseconds(start);

    // Scale OMP iterations by type calculation complexity
    double factor;

    // if tuning_weight_scale_ is a number that looks valid, use it as the factor
    if (OperatorTuneBase::tuning_weight_scale_ > 0.01) {
      factor = OperatorTuneBase::tuning_weight_scale_;
    } else {
      // These are empirically-determined constants found by balancing between
      // a desktop (8 & 12 cpu's) and large cloud instances (32 & 64 cpu's)
      switch (mshadow::DataType<DType>::kFlag) {
        case mshadow::kUint8:
        case mshadow::kInt8:
          factor = 8.5;
          break;
        case mshadow::kInt32:
          factor = 4.5;
          break;
        case mshadow::kInt64:
          factor = 2;
          break;
        case mshadow::kFloat64:
          factor = 1.25;
          break;
        case mshadow::kFloat32:
        default:
          factor = 1.0;
          break;
      }
    }

    wl_count = static_cast<int>(factor * OperatorTuneBase::WORKLOAD_COUNT * omp_thread_count);
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < OUTSIDE_COUNT; ++i) {
      #pragma omp parallel for num_threads(omp_thread_count)
      for (int x = 0; x < wl_count; ++x) {
        // trivial operation
        volatile_int_ += x;
      }
    }
    const duration_t omp_duration = OperatorTuneBase::GetDurationInNanoseconds(start)
                                    - no_omp_duration;
    return omp_duration >> OUTSIDE_COUNT_SHIFT;
  }

  /*! \brief Measure OMP overhead for a trivial OMP loop using all cores
   * \returns Time in nanoseconds to initialize/cleanup when excuting an OMP block
   */
  static duration_t GetOMPLoopOverhead() {
    // It was found empirically that OMP times was not heavily tied to number of cores,
    // so take an average across all core counts
    const auto max_cores_default = static_cast<size_t>(omp_get_num_procs()) >> 1;
    const auto max_cores = dmlc::GetEnv("MXNET_USE_NUM_CORES_OPERATOR_TUNING", max_cores_default);
    if (max_cores >= 2) {
      std::vector<duration_t> core_times;
      // Take care of any OMP lazy-init with a throwaway call
      for (size_t omp_threads = 2; omp_threads <= max_cores; ++omp_threads) {
        GetOMPLoopOverhead(omp_threads);
      }
      std::vector<duration_t> durations;
      durations.reserve(max_cores - 1);
      for (size_t omp_threads = 2; omp_threads <= max_cores; ++omp_threads) {
        const duration_t duration = GetOMPLoopOverhead(omp_threads);
        if (OperatorTuneBase::verbose_tuning_info_) {
          LOG(INFO) << "OMP Thread Count: " << omp_threads << ", overhead: " << duration << " ns";
        }
        durations.emplace_back(duration);
      }
      // return median
      std::sort(durations.begin(), durations.end());
      return durations[durations.size() >> 1];
    }
    return INT_MAX;  // If only one core, then never use OMP (say the overhead is huge)
  }

  /*!
   * \brief Some string utility functions that aren't specific to tuning
   */
  struct StringUtil {
    /*!
     * \brief Terim whitespace from beninning and end of string
     * \param s String to trimp
     * \return reference to the modified string. This is the same std::string object as what was
     *         supplied in the parameters
     */
    static std::string &trim(std::string *s) {
      s->erase(s->begin(), std::find_if(s->begin(), s->end(), [](int ch) {
        return !std::isspace(ch);
      }));
      s->erase(std::find_if(s->rbegin(), s->rend(), [](int ch) {
        return !std::isspace(ch);
      }).base(), s->end());
      return *s;
    }

    /*!
     * \brief Tokenize a string into a list of tokens
     * \param s String to tokenize
     * \return std::list of tokens
     */
    static std::list<std::string> string2list(const std::string &s) {
      std::list<std::string> res;
      std::istringstream iss(s);
      std::string token;
      while (std::getline(iss, token, ',')) {
        trim(&token);
        if (!token.empty()) {
          res.push_back(token);
        }
      }
      return res;
    }
  };

  /*!
   * \brief Get data type from string representation
   * \warning Do not call from a performance-sensitive area
   */
  static int type_from_string(const std::string& type_string) {
    if (type_string == "float32")
      return mshadow::kFloat32;
    if (type_string == "float64")
      return mshadow::kFloat64;
    if (type_string == "float16")
      return mshadow::kFloat16;
    if (type_string == "bfloat16")
      return mshadow::kBfloat16;
    if (type_string == "int8")
      return mshadow::kInt8;
    if (type_string == "uint8")
      return mshadow::kUint8;
    if (type_string == "int16")
      return mshadow::kInt16;
    if (type_string == "uint16")
      return mshadow::kUint16;
    if (type_string == "int32")
      return mshadow::kInt32;
    if (type_string == "int64")
      return mshadow::kInt64;
    return -1;  // invalid
  }

  /*!
   * \brief Parse MXNET_USE_OPERATOR_TUNING environment variable
   * \param config String representation of MXNET_ENABLE_OPERATOR_TUNING environment variable
   *        Values:
   *            0=disable all
   *            1=enable all
   *            float32, float16, float32=list of types to enable, and disable those not listed
   */
  static void ParseEnablerConfig(std::string config) {
    StringUtil::trim(&config);
    if (!config.empty()) {
      // First disable all
      OperatorTuneByType<float>::set_tuning_mode(tune::kAlwaysOMP);
      OperatorTuneByType<double>::set_tuning_mode(tune::kAlwaysOMP);
      OperatorTuneByType<int8_t>::set_tuning_mode(tune::kAlwaysOMP);
      OperatorTuneByType<uint8_t>::set_tuning_mode(tune::kAlwaysOMP);
      OperatorTuneByType<int16_t>::set_tuning_mode(tune::kAlwaysOMP);
      OperatorTuneByType<uint16_t>::set_tuning_mode(tune::kAlwaysOMP);
      OperatorTuneByType<int32_t>::set_tuning_mode(tune::kAlwaysOMP);
      OperatorTuneByType<int64_t>::set_tuning_mode(tune::kAlwaysOMP);
      // See if it's a non-number (ie type or list of types)
      if (!::isdigit(config[0])) {
        OperatorTuneByType<mshadow::half::half_t>::set_tuning_mode(tune::kAuto);
        std::list<std::string> tokens = StringUtil::string2list(config);
        for (const std::string& stype : tokens) {
          // We don't have an enum for halt_t
          const int typ = type_from_string(stype);
          if (typ >= 0) {
            switch (typ) {
              case mshadow::kFloat32:
                OperatorTuneByType<float>::set_tuning_mode(tune::kAuto);
                break;
              case mshadow::kFloat64:
                OperatorTuneByType<double>::set_tuning_mode(tune::kAuto);
                break;
              case mshadow::kFloat16:
                OperatorTuneByType<mshadow::half::half_t>::set_tuning_mode(tune::kAuto);
                break;
              case mshadow::kInt8:
                OperatorTuneByType<int8_t>::set_tuning_mode(tune::kAuto);
                break;
              case mshadow::kUint8:
                OperatorTuneByType<uint8_t>::set_tuning_mode(tune::kAuto);
                break;
              case mshadow::kInt16:
                OperatorTuneByType<int16_t>::set_tuning_mode(tune::kAuto);
                break;
              case mshadow::kUint16:
                OperatorTuneByType<uint16_t>::set_tuning_mode(tune::kAuto);
                break;
              case mshadow::kInt32:
                OperatorTuneByType<int32_t>::set_tuning_mode(tune::kAuto);
                break;
              case mshadow::kInt64:
                OperatorTuneByType<int64_t>::set_tuning_mode(tune::kAuto);
                break;
              default:
                CHECK(false) << "Unsupported tuning data type: " << stype;
                break;
            }
          } else {
            // -1 is error
            LOG(WARNING) << "Unknown data type to be tuned: " << stype;
          }
        }
      } else {
        if (std::atoi(config.c_str()) > 0) {
          OperatorTuneByType<float>::set_tuning_mode(tune::kAuto);
          OperatorTuneByType<double>::set_tuning_mode(tune::kAuto);
          OperatorTuneByType<int8_t>::set_tuning_mode(tune::kAuto);
          OperatorTuneByType<uint8_t>::set_tuning_mode(tune::kAuto);
          OperatorTuneByType<int16_t>::set_tuning_mode(tune::kAuto);
          OperatorTuneByType<uint16_t>::set_tuning_mode(tune::kAuto);
          OperatorTuneByType<int32_t>::set_tuning_mode(tune::kAuto);
          OperatorTuneByType<int64_t>::set_tuning_mode(tune::kAuto);
          OperatorTuneByType<mshadow::half::half_t>::set_tuning_mode(tune::kAuto);
        }
      }
    }
  }

  /*! \brief Whether this object has been initialized */
  static bool initialized_;
  /*! \brief Number of passes to obtain an average */
  static constexpr duration_t OUTSIDE_COUNT = (1 << OUTSIDE_COUNT_SHIFT);
  /*! \brief Random data for timing operator calls */
  static std::unique_ptr<DType[]> data_set_;
  /*! \brief Operators tuned */
  static std::unordered_set<std::string> operator_names_;
  /*! \brief Arbitary object to modify in OMP loop */
  static volatile int volatile_int_;
  /*! \brief Output insertable (into code) instantiation+default-value macros */
  static bool output_tuning_data_;
};

// TODO(leezu) uncomment once gcc4.8 support is dropped; fixes Wno-undefined-var-template
// template <typename DType>
// std::unordered_set<std::string> OperatorTune<DType>::operator_names_;

/*!
 * \brief Class that tunes unary operators
 * \tparam DType Data type to be used when tuning the kernel operations
 */
template<typename DType>
class UnaryOpTune : public OperatorTune<DType> {
 protected:
  typedef OperatorTune<DType> Super;
  using duration_t = typename Super::duration_t;
  using Tick = typename Super::Tick;

  /*!
   * \brief Determine the time it takes a kernel operator to execute WORKLOAD_COUNT iterations
   *        Used for kernels that take no arguments (ie set_zero)
   * \tparam OP Kernel operator
   * \return Duration in nanoseconds for the 'WORKLOAD_COUNT' operations
   */
  template<typename OP>
  static duration_t GetBlankWorkload() {
    DType tmp;
    volatile DType *res = &tmp;
    const Tick start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < Super::WORKLOAD_COUNT; ++i) {
      // Use a logical AND instead of mod to avoid affecting the timing result with a slow divide
      *res += OP::Map();
    }
    const duration_t omp_duration = Super::GetDurationInNanoseconds(start);
    return omp_duration ? omp_duration : 1;
  }

  /*!
   * \brief Determine the time it takes a kernel operator to execute WORKLOAD_COUNT iterations
   *        Used for kernels that take one argument (ie sqrt())
   * \tparam OP Kernel operator
   * \return Duration in nanoseconds for the 'WORKLOAD_COUNT' operations
   */
  template<typename OP>
  static duration_t GetUnaryWorkload() {
    DType tmp;
    volatile DType *res = &tmp;
    const Tick start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < Super::WORKLOAD_COUNT; ++i) {
      // Use a logical AND instead of mod to avoid affecting the timing result with a slow divide
      *res = OP::Map(Super::data_set_[i & 0xFF]);
    }
    const duration_t omp_duration = Super::GetDurationInNanoseconds(start);
    return omp_duration ? omp_duration : 1;
  }

  /*!
   * \brief Determine the time it takes a kernel operator to execute WORKLOAD_COUNT iterations
   *        Used for kernels that take two arguments (ie elemwise_add())
   * \tparam OP Kernel operator
   * \return Duration in nanoseconds for the 'WORKLOAD_COUNT' operations
   */
  template<typename OP>
  static inline duration_t GetBinaryWorkload() {
    DType tmp;
    volatile DType *res = &tmp;
    const Tick start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < Super::WORKLOAD_COUNT; ++i) {
      // Use a logical AND instead of mod to avoid affecting the timing result with a slow divide
      *res = OP::Map(Super::data_set_[i & 0xFF], Super::data_set_[(i + 1) & 0xFF]);
    }
    const duration_t omp_duration = Super::GetDurationInNanoseconds(start);
    return omp_duration ? omp_duration : 1;
  }

  /*!
   * \brief Determine the time it takes a kernel operator to execute WORKLOAD_COUNT iterations
   *        Used for kernels that take three arguments (ie backwards_grad<elemwise_add>())
   * \tparam OP Kernel operator
   * \return Duration in nanoseconds for the 'WORKLOAD_COUNT' operations
   */
  template<typename OP>
  static duration_t GetTertiaryWorkload() {
    DType tmp;
    volatile DType *res = &tmp;
    const Tick start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < Super::WORKLOAD_COUNT; ++i) {
      // Use a logical AND instead of mod to avoid affecting the timing result with a slow divide
      *res = OP::Map(Super::data_set_[i & 0xFF],
                     Super::data_set_[(i + 1) & 0xFF],
                     Super::data_set_[i & 0xFF]);
    }
    const duration_t omp_duration = Super::GetDurationInNanoseconds(start);
    return omp_duration ? omp_duration : 1;
  }

  /*!
   * \brief Determine the time it takes a kernel operator to execute WORKLOAD_COUNT iterations
   *        Used for mxnet-like kernels that take no arguments)
   * \tparam OP Kernel operator
   * \return Duration in nanoseconds for the 'WORKLOAD_COUNT' operations
   */
  template<typename OP>
  static duration_t GetBlankWorkloadEx() {
    std::unique_ptr<DType[]> tmp(new DType[Super::WORKLOAD_COUNT]);
    DType *tmp_ptr = tmp.get();
    const Tick start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < Super::WORKLOAD_COUNT; ++i) {
      OP::Map(i, tmp_ptr);
    }
    const duration_t omp_duration = Super::GetDurationInNanoseconds(start);
    return omp_duration ? omp_duration : 1;
  }

 public:
  /*!
   * \brief Tune the specified kernel operator.  Optionally print out C++ macro that defines the
   *        tuning data variable and the default tuned value
   *        This function tunes an operator which takes no arguments
   * \tparam OP The kernel operator to be tuned
   */
  template<typename OP>
  static void TuneBlankOperator() {
    mxnet::op::mxnet_op::tuned_op<OP, DType>::workload_[0] = GetBlankWorkload<OP>();
    if (Super::output_tuning_data_) {
      std::cout << "IMPLEMENT_UNARY_WORKLOAD_FWD("
                << Super::template type_name<OP>()
                << ");  // NOLINT()" << std::endl << std::flush;  // For long lines
    }
  }

  /*!
   * \brief Tune the specified kernel operator.  Optionally print out C++ macro that defines the
   *        tuning data variable and the default tuned value
   *        This function tunes an operator which takes one argument
   * \tparam OP The kernel operator to be tuned
   */
  template<typename OP>
  static void TuneUnaryOperator() {
    mxnet::op::mxnet_op::tuned_op<OP, DType>::workload_[0] = GetUnaryWorkload<OP>();
    if (Super::output_tuning_data_) {
      std::cout << "IMPLEMENT_UNARY_WORKLOAD_FWD("
                << Super::template type_name<OP>()
                << ");  // NOLINT()" << std::endl << std::flush;  // For long lines
    }
  }

  /*!
   * \brief Tune the specified kernel operator.  Optionally print out C++ macro that defines the
   *        tuning data variable and the default tuned value
   *        This function tunes a backward operator which takes one argument
   * \tparam OP The kernel operator to be tuned
   */
  template<typename OP>
  static void TuneUnaryBackwardOperator() {
    mxnet::op::mxnet_op::tuned_op<mxnet_op::backward_grad_tuned<OP>, DType>::workload_[0] =
      GetBinaryWorkload<mxnet::op::mxnet_op::backward_grad_tuned<OP>>();
    if (Super::output_tuning_data_) {
      std::cout << "IMPLEMENT_UNARY_WORKLOAD_BWD("
                << Super::template type_name<OP>()
                << ");  // NOLINT()" << std::endl << std::flush;  // For long lines
    }
  }

  /*!
   * \brief Tune the specified "mxnet_op-type" kernel operator.
   *        Optionally print out C++ macro that defines the
   *        tuning data variable and the default tuned value
   *        This function tunes an operator which takes no arguments
   * \tparam OP The kernel operator to be tuned
   */
  template<typename OP>
  static void TuneBlankOperatorEx() {
    mxnet::op::mxnet_op::tuned_op<OP, DType>::workload_[0] = GetBlankWorkloadEx<OP>();
    if (Super::output_tuning_data_) {
      std::cout << "IMPLEMENT_BLANK_WORKLOAD_FWD("
                << Super::template type_name<OP>()
                << ");  // NOLINT()" << std::endl << std::flush;  // For long lines
    }
  }

  /*!
   * \brief Determine whether to use OMP based upon both timing and configuration using the
   *        given (templated) operator's workload
   * \tparam OP Operator whose workload to use (tuned_op::workload_[0])
   * \param N Number of iterations desired
   * \param thread_count Number of OMP threads available to perform the iterations
   * \returns Whether it's faster to use OMP for these iterations
   */
  template<typename OP>
  inline static bool UseOMP(size_t N, size_t thread_count) {
      return OperatorTune<DType>::UseOMP(N,
                                         thread_count,
                                         static_cast<uint64_t>(N) * OP::workload_[0]);
  }
};

/*!
 * \brief Class that tunes binary and unary operators
 * \tparam DType Data type to be used when tuning the kernel operations
 */
template<typename DType>
class BinaryOpTune : public UnaryOpTune<DType> {
 protected:
  typedef UnaryOpTune<DType> Super;

 public:
  /*!
   * \brief Tune a generic binary operator
   * @tparam OP - Operator type
   */
  template<typename OP>
  static void TuneBinaryOperator() {
    mxnet_op::tuned_op<OP, DType>::workload_[0] = Super::template GetBinaryWorkload<OP>();
    if (Super::Super::output_tuning_data_) {
      std::cout << "IMPLEMENT_BINARY_WORKLOAD_FWD("
                << Super::template type_name<OP>()
                << ");  // NOLINT()" << std::endl << std::flush;  // For long lines
    }
  }

  /*!
   * \brief Tune binary backward operator
   * \tparam OP - operator
   */
  template<typename OP>
  static void TuneBinaryBackwardOperator() {
    mxnet::op::mxnet_op::tuned_op<mxnet_op::backward_grad_tuned<OP>, DType>::workload_[0] =
      Super::template GetTertiaryWorkload<mxnet::op::mxnet_op::backward_grad_tuned<OP>>();
    if (Super::Super::output_tuning_data_) {
      std::cout << "IMPLEMENT_BINARY_WORKLOAD_BWD("
                << Super::template type_name<OP>()
                << ");  // NOLINT()" << std::endl << std::flush;  // For long lines
    }
  }
};

#undef OUTSIDE_COUNT_SHIFT
#undef WORKLOAD_COUNT_SHIFT

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_OPERATOR_TUNE_INL_H_
