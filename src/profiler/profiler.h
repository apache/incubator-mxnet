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
 * Copyright (c) 2015 by Contributors
 * \file profiler.h
 * \brief implements profiler
 */
#ifndef MXNET_PROFILER_PROFILER_H_
#define MXNET_PROFILER_PROFILER_H_

#include <dmlc/concurrentqueue.h>
#include <dmlc/thread_group.h>
#include <vector>
#include <string>
#include <cstdint>
#include <mutex>
#include <memory>
#include <array>
#include "./vtune.h"
#include "./aggregate_stats.h"
#include "../common/nvtx.h"
#include "../common/utils.h"


namespace mxnet {
namespace profiler {



/*!
 * \brief Constant-sized character array class with simple string API to avoid allocations
 * \tparam string_size Maximum size of the string (including zero-terminator)
 */
template<size_t string_size>
struct static_string {
  inline static_string() { string_[0] = '\0'; }
  inline explicit static_string(const char *s) { set(s); }
  inline const char *c_str() const { return &string_[0]; }
  inline void set(const char *s) {
#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
    strncpy(&string_[0], s, string_size - 1);
#pragma GCC diagnostic pop
    string_[string_size - 1] = '\0';
  }
  inline void append(const char *s) {
    const size_t l = strlen(&string_[0]);
    if (l < string_size - 1) {
      strncpy(&string_[0] + l, s, string_size - l - 1);
      string_[string_size - 1] = '\0';
    }
  }
 private:
  /*! \brief The actual character array */
  std::array<char, string_size> string_;
};

using profile_stat_string = static_string<128>;

/*!
 * \brief Base profile statistic structure
 */
struct ProfileStat {
  /*!
   * \brief Event type as used for chrome://tracing support
   * \note Tracing formats: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview NOLINT(*)
   */
  enum EventType {
    kDurationBegin = 'B',
    kDurationEnd = 'E',
    kComplete = 'X',
    kInstant = 'i',
    kCounter = 'C',
    kAsyncNestableStart = 'b',
    kAsyncNestableInstant = 'n',
    kAsyncNestableEnd = 'e',
    kFlowStart = 's',
    kFlowStep = 't',
    kFlowEnd = 'f',
    kSample = 'P',
    kObjectCreated = 'N',
    kObjectSnapshot = 'O',
    kObjectDestroyed = 'D',
    kMetadata = 'M',
    kMemoryDumpGlobal = 'V',
    kMemoryDumpProcess = 'v',
    kMark = 'R',
    kClockSync = 'c',
    kContextEnter = '(',
    kContextLeave = ')'
  };

  struct SubEvent {
    /*! \brief whether this sub-event object is enabled */
    bool enabled_ = false;
    /*! \brief Type of the sub-event */
    EventType event_type_;
    /*! \brief Timestamp of sub-event */
    uint64_t timestamp_;
  };

  /*! \brief operation name */
  profile_stat_string name_;

  /*! \brief operation categories (comma-delimited) */
  profile_stat_string categories_;

  /*! \brief whether to add this stat to AggregateStats */
  bool enable_aggregate_ = true;

  /* !\brief Process id */
  size_t process_id_ = common::current_process_id();

  /*! \brief id of thread which operation run on.
   *
   * */
  std::thread::id thread_id_ = std::this_thread::get_id();  // Not yet seen a
                                                            // case where this isn't valid

  /*! \brief Sub-events (ie begin, end, etc.) */
  SubEvent items_[3];  // Don't use vector in order to avoid memory allocation

  /*!
   * \brief Get current tick count in microseconds
   * \return Current arbitrary tick count in microseconds
   */
  static inline uint64_t NowInMicrosec() {
#if defined(_MSC_VER) && _MSC_VER <= 1800
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return counter.QuadPart * 1000000 / frequency.QuadPart;
#else
    return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
  }

  /*!
   * \brief Print event statistics in json format to the supplied output stream
   * \param os Output stream to write the data
   * \note Emits all sub-even statistics
   */
  void EmitEvents(std::ostream *os) {
    size_t count = 0;
    for (size_t i = 0; i < sizeof(items_) / sizeof(items_[0]); ++i) {
      if (items_[i].enabled_) {
        if (count) {
          *os << ",\n";
        }
        EmitSubEvent(os, i);
        ++count;
      }
    }
  }

  /*!
   * \brief Virtual destructor
   */
  virtual ~ProfileStat() {}

  /*!
   * \brief Save aggregate data for this stat
   * \param data Stat data
   */
  virtual void SaveAggregate(AggregateStats::StatData *data) const {
    if (data) {
      data->type_ = AggregateStats::StatData::kOther;
    }
  }

 protected:
  /*!
   * \brief Override to emit extra items within the json event data block. Append with a comma ",".
   * \param os Output stream to write data to
   * \param idx Sub-even index (index into items_) to write
   */
  virtual void EmitExtra(std::ostream *os, size_t idx) {}

  /*!
   * \brief Emit sub-event statistics
   * \param os Output stream
   * \param idx Sub-even index (index into items_) to write
   */
  void EmitSubEvent(std::ostream *os, size_t idx) {
    const SubEvent &ev = items_[idx];
    if (ev.enabled_) {
      *os << "    {\n"
          << "        \"name\": \"" << name_.c_str() << "\",\n"
          << "        \"cat\": " << "\"" << categories_.c_str() << "\",\n"
          << "        \"ph\": \"" << static_cast<char>(ev.event_type_) << "\",\n"
          << "        \"ts\": " << ev.timestamp_ << ",\n";
      EmitExtra(os, idx);
      *os << "        \"pid\": " << process_id_ << ",\n"
          << "        \"tid\": " << std::hash<std::thread::id>{}(thread_id_) << "\n"
          << "    }\n";
    }
  }
};

/*!
 * \brief Device statistics
 */
struct DeviceStats {
  using TQueue = dmlc::moodycamel::ConcurrentQueue<ProfileStat *>;
  /*!
   * \brief Destructor, clean up allocated objects
   */
  ~DeviceStats() {
    std::shared_ptr<TQueue> es = opr_exec_stats_;
    if (es) {
      ProfileStat *stat = nullptr;
      while (es->try_dequeue(stat)) {
        delete stat;
      }
    }
  }

  /*! \brief device name */
  std::string dev_name_;
  /*! \brief operation execution statistics on this device */
  std::shared_ptr<TQueue> opr_exec_stats_ = std::make_shared<TQueue>();
};

/*!
 *  _____              __  _  _
 * |  __ \            / _|(_)| |
 * | |__) |_ __  ___ | |_  _ | | ___  _ __
 * |  ___/| '__|/ _ \|  _|| || |/ _ \| '__|
 * | |    | |  | (_) | |  | || |  __/| |
 * |_|    |_|   \___/|_|  |_||_|\___||_|
 *
 * \brief profiler that records the operation execution information
 *        and saves the profile statistics.
 * \note Profiler class doesn't know anything about VTune
 */
class Profiler {
 public:
  enum ProfilerMode {
      kSymbolic = 1,
      kImperative = 2,
      kAPI = 4,
      kMemory = 8
  };
  enum ProfilerState {
      kNotRunning = 0,
      kRunning = 1
  };

  /*! \brief set state of profiler */
  void SetState(ProfilerState state);
  /*! \return state of profiler */
  inline ProfilerState GetState() const {
    return this->state_;
  }
  /*!
   * \brief set profiler configuration
   * \param mode flags, one or more of 'ProfilerMode'
   * \param output_filename profile output file name
   * \param continuous_dump true if profile information should be periodically dumped
   * \param dump_period Period (in seconds) of profile info dumping
   */
  void SetConfig(int mode, std::string output_filename,
                 bool continuous_dump,
                 float dump_period,
                 bool aggregate_stats);

  /*! \return mode of profiler */
  inline int GetMode() const {
    return this->mode_;
  }

  inline bool IsProfiling(const ProfilerMode pm) const {
    return GetState() == kRunning && (GetMode() & pm) == pm;
  }

  /*! \return whether the profiler is enabled to output */
  inline bool IsEnableOutput() const {
    return this->enable_output_;
  }
  /*!
   * \brief dump the profile file
   * \param perform_cleanup Close off the json trace structures (ie last pass)
   */
  void DumpProfile(bool perform_cleanup = true);

  /*! \return the profiler init time, time unit is microsecond (10^-6) s */
  uint64_t MSHADOW_CINLINE GetInitTime() const {
    return init_time_;
  }
  /*!
   * \brief add one operation execution record in corresponding device statistics
   * \tparam SetExtraInfoFunction
   * \param dev_type
   * \param dev_id
   * \param set_extra_info_function
   * \note Because when this function exits, the object is written to the profile queue,
   *       and at that point, could be consumed and/or destroyed at any moment,
   *       any preprocessing on the object is to be done in the set_extra_info_function
   *       callback.  Another option is to use the CreateProfileStat()/AddProfileStat() pair,
   *       adding it only after
   */
  template<typename StatType, typename SetExtraInfoFunction, typename ...Args>
  void AddNewProfileStat(SetExtraInfoFunction set_extra_info_function, Args... args) {
    if (!paused_) {
      std::unique_ptr<StatType> stat = CreateProfileStat<StatType>(args...);
      set_extra_info_function(stat.get());
      AddProfileStat(&stat);
    }
  }

  /*!
   * \brief Return aggregate statistic accumulator
   * \return shared pointer to the 'ProfileStats' aggregate statistic accumulator
   */
  std::shared_ptr<AggregateStats> GetAggregateStats() const {
    return aggregate_stats_;
  }

  /*!
   * \brief Get a pointer to the Profiler singleton
   * \return Profiler singleton
   * \param sp Profiler shared pointer, only use for singleton ownership
   */
  static Profiler* Get(std::shared_ptr<Profiler> *sp = nullptr);

  /*!
   * \brief Set whether statistic collection is to be paused
   * \param paused true if statistic collection is to be paused, otherwise
   * resume statistic collection
   * \note Pause/Resume is not recursive
   */
  void set_paused(bool paused) { paused_ = paused; }

  /*!
   * \brief Get the calculated device count (numb er of devices to track in profile data).
   * \return Device count
   * \note Number of CPU's + Number of GPU's + One for CPU-Pinned
   */
  size_t DeviceCount() const { return cpu_num_ + gpu_num_ + 2; }

  /*!
   * \brief Compute device index given device type and id
   * \param dev_type Device type
   * \param dev_id Device ID
   * \return Device index for indexing into device-specific data
   */
  size_t DeviceIndex(mxnet::Context::DeviceType dev_type, int32_t dev_id);

  /*!
   * \brief Device name
   * \param dev_type Device type
   * \param dev_id Device ID
   * \return Character pointer to device name
   */
  const char *DeviceName(mxnet::Context::DeviceType dev_type, int32_t dev_id);


  /*!
   * \brief Device name
   * \param dev_type Device type
   * \param dev_id Device ID
   * \return Character pointer to device name
   */
  const char *DeviceName(const size_t index);

  /*!
   * \brief Whether aggregate stats are being collected
   * \return true if aggregate stats are being collected
   */
  inline bool AggregateEnabled() const {
    return aggregate_stats_.get() != nullptr;
  }

  /*!
   * \brief Whether aggregate stats are currently being recorded
   * \return true if aggregate stats are currently being recorded
   */
  inline bool AggregateRunning() const {
    return GetState() == kRunning && AggregateEnabled();
  }

 public:
  /*!
   * \brief Constructor
   */
  Profiler();

  /*!
   * \brief Destructor
   */
  virtual ~Profiler();

 private:
  /*!
   * \brief Create a new profile statistic object
   * \tparam StatType The type of the profile statistic object
   * \tparam Args Argument types to pass to the new object's constructor
   * \param args Arguments to pass to the new object's constructor
   * \return A unique_ptr to the new statistic object
   */
  template<typename StatType, typename ...Args>
  static std::unique_ptr<typename std::enable_if<std::is_base_of<ProfileStat, StatType>::value,
    StatType>::type> CreateProfileStat(Args... args) {
    return std::unique_ptr<StatType>(new StatType(args...));
  }

  /*!
   * \brief Add a general profile statistic object
   * \tparam StatType Type of the statistic object
   * \param stat The statistic object
   */
  template<typename StatType>
  inline void AddProfileStat(std::unique_ptr<StatType> *stat) {
    general_stats_.opr_exec_stats_->enqueue(stat->release());
  }

  /*! \brief generate device information following chrome profile file format */
  void EmitPid(std::ostream *os, const std::string& name, size_t pid);

  /*!
   * \brief Set continuous asynchronous profile dump
   * \param continuous_dump Whether to continuously dump profile information
   * \param delay_in_seconds Delay between asynchronous dumps
   */
  void SetContinuousProfileDump(bool continuous_dump, float delay_in_seconds);

  /*! \brief internal mutex of the profiler */
  std::recursive_mutex m_;
  /*! \brief indicate whether the profiler is running */
  ProfilerState state_;
  /*! \brief once running, enable profiler to output */
  volatile bool enable_output_;
  /*! \brief indicate what operator the profiler will record */
  int mode_ = kSymbolic | kAPI | kMemory;
  /*! \brief filename to output profile file */
  std::string filename_ = "profile.json";
  /*! \brief profile statistics consist of multiple device statistics */
  std::unique_ptr<DeviceStats[]> profile_stat;
  /*! \brief Stats not associated directly with a device */
  DeviceStats  general_stats_;
  /*! \brief Map category -> pid */
  std::unordered_map<std::string, size_t> category_to_pid_;
  /*! \brief cpu number on the machine */
  unsigned int cpu_num_;
  /*! \brief gpu number on the machine */
  unsigned int gpu_num_;
  /*! \brief the profiler init time */
  uint64_t init_time_;
  /*! \brief Continuously dump profile info */
  volatile bool continuous_dump_ = false;
  /*! \brief Number of non-meta profiling record emitted */
  volatile uint64_t num_records_emitted_ = 0;
  /*! \brief Number of times profile was dumped */
  volatile uint64_t profile_dump_count_;
  /*! \brief Whether profiling is paused */
  volatile bool paused_ = false;
  /*! \brief Maintain in-memory aggregate stats for print output.
   *  \warning This has a negative performance impact */
  std::shared_ptr<AggregateStats> aggregate_stats_ = nullptr;
  /*! \brief Asynchronous operation thread lifecycle control object */
  std::shared_ptr<dmlc::ThreadGroup> thread_group_ = std::make_shared<dmlc::ThreadGroup>();
  /* !\brief pids */
  std::unordered_set<uint32_t> process_ids_;
};

#ifdef MXNET_USE_VTUNE
#define VTUNE_ONLY_CODE(...) __VA_ARGS__  /* This is undefined at the bottom of this file */
#else
#define VTUNE_ONLY_CODE(...) /* */        /* This is undefined at the bottom of this file */
#endif

#ifdef MXNET_USE_NVTX
#define NVTX_ONLY_CODE(...) __VA_ARGS__  /* This is undefined at the bottom of this file */
#else
#define NVTX_ONLY_CODE(...) /* */        /* This is undefined at the bottom of this file */
#endif

/**
 *  _____              __  _  _  _                ____  _     _            _
 * |  __ \            / _|(_)| |(_)              / __ \| |   (_)          | |
 * | |__) |_ __  ___ | |_  _ | | _ _ __   __ _  | |  | | |__  _  ___   ___| |_  ___
 * |  ___/| '__|/ _ \|  _|| || || | '_ \ / _` | | |  | | '_ \| |/ _ \ / __| __|/ __|
 * | |    | |  | (_) | |  | || || | | | | (_| | | |__| | |_) | |  __/| (__| |_ \__ \
 * |_|    |_|   \___/|_|  |_||_||_|_| |_|\__, |  \____/|_.__/| |\___| \___|\__||___/
 *                                        __/ |             _/ |
 *                                       |___/             |__/
 */

enum ProfileObjectType {
  kDomain,
  kCounter,
  kTask,
  kEvent,
  kFrame
};

class ProfileObject {
 public:
  /*!
   * \brief Virtual destructor for child classes
   */
  virtual ~ProfileObject() {}
  /*!
   * \brief Return profiling object object type (i.e. kTask, kEvent, ...)
   * \return Profiling object type
   */
  virtual ProfileObjectType type() const = 0;
};

/*!
 * \brief Tuning domain. Used in VTune to separate similar tasks/counters/etc. (for example).
 *        For chrome tracing, generally maps to category.
 */
struct ProfileDomain : public ProfileObject {
  /*!:(
   * \brief Constructor
   * \param name Name of the domain
   */
  explicit ProfileDomain(const char *name) noexcept
    : name_(name) {
    CHECK_NOTNULL(name);
    CHECK_NE(name[0], '\0');
    VTUNE_ONLY_CODE(vtune_domain_.reset(new vtune::VTuneDomain(name)));
  }
  /*!
   * \brief Get domain name
   * \return Domain name
   */
  const char *name() const { return name_.c_str(); }
  ProfileObjectType type() const override { return kDomain; }
  VTUNE_ONLY_CODE(inline vtune::VTuneDomain *dom() { return vtune_domain_.get(); });
 private:
  /*! \brief Name of the domain */
  profile_stat_string name_;
  /*! \brief VTune domain object */
  VTUNE_ONLY_CODE(std::unique_ptr<vtune::VTuneDomain> vtune_domain_);
};

/*!
 * \brief Counter object and statistic item
 */
struct ProfileCounter : public ProfileObject {
  /*!
   * \brief Constructor
   * \param name Counter name
   * \param domain Counter domain
   */
  ProfileCounter(const char *name, ProfileDomain *domain) noexcept
    : name_(name)
      , domain_(domain)
      , value_(0) {
    CHECK_NOTNULL(domain);
    VTUNE_ONLY_CODE(vtune_.reset(new vtune::VTuneCounter(name, domain->dom())));
  }
  ~ProfileCounter() {}
  /*! \brief operator: ++object */
  inline uint64_t operator ++() {
    return IncrementValue(1);
  }
  /*! \brief operator: object++ */
  inline uint64_t operator ++(int) {
    const uint64_t old = value_;
    IncrementValue(1);
    return old;
  }
  /*! \brief operator: --object */
  inline uint64_t operator --() {
    CHECK_GT(value_, 0);
    return DecrementValue(1);
  }
  /*! \brief operator: object-- */
  inline uint64_t operator --(int) {
    CHECK_GT(value_, 0);
    const uint64_t old = value_;
    DecrementValue(1);
    return old;
  }
  /*! \brief operator: object += v */
  inline uint64_t operator +=(int64_t v) {
    if (v >= 0) {
      return IncrementValue(static_cast<uint64_t>(v));
    } else {
      v = -v;
      return DecrementValue(static_cast<uint64_t>(v));
    }
  }
  /*! \brief operator: object -= v */
  inline uint64_t operator -=(int64_t v) {
    CHECK_GE(value_, static_cast<uint64_t>(v));
    if (v >= 0) {
      return DecrementValue(static_cast<uint64_t>(v));
    } else {
      v = -v;
      return IncrementValue(static_cast<uint64_t>(v));
    }
  }

  inline bool operator >=(int64_t v) {
      CHECK_GE(v, 0);
      return value_ >= static_cast<uint64_t>(v);
  }

  /*! \brief operator: object = v */
  inline ProfileCounter& operator = (uint64_t v) {
    SetValue(v);
    return *this;
  }

  ProfileObjectType type() const override { return kCounter; }

 protected:
  /*!
   * \brief Count statistic object
   */
  struct ProfileCounterStat : public ProfileStat {
    uint64_t value_;
    explicit ProfileCounterStat(const char *name, uint64_t value) : value_(value) {
      items_[0].enabled_ = true;
      items_[0].event_type_ = kCounter;
      items_->timestamp_ = NowInMicrosec();
      name_.set(name);
    }

    /*!
     * \brief Emit counter value as extra data for this statistic type
     * \param os Output stream to write data to
     * \param idx Sub-even index (index into items_) to write
     */
    void EmitExtra(std::ostream *os, size_t idx) override {
      ProfileStat::EmitExtra(os, idx);
      *os << "        \"args\": { \"" << name_.c_str() << "\": " << value_ << " },\n";
    }

    /*!
     * \brief Save aggregate data for this stat
     * \param data Stat data
     */
    void SaveAggregate(AggregateStats::StatData *data) const override {
      if (data) {
        data->type_ = AggregateStats::StatData::kCounter;
        ++data->total_count_;
        data->total_aggregate_ = value_;
        if (value_ > data->max_aggregate_) {
          data->max_aggregate_ = value_;
        }
        if (value_ < data->min_aggregate_) {
          data->min_aggregate_ = value_;
        }
      }
    }
  };

 private:
  /*!
   * \brief Send this object's statistical datapoint to the profiler
   */
  inline void SendStat(uint64_t value) {
    Profiler::Get()->AddNewProfileStat<ProfileCounterStat>([this](ProfileCounterStat *stat) {
                                                             stat->categories_.set(domain_->name());
                                                           },
                                                           name_.c_str(),
                                                           value);
  }

  /*!
   * \brief Set counter value
   * \param val Value to set the counter
   */
  inline void SetValue(uint64_t val) {
    VTUNE_ONLY_CODE(*vtune_ = val);
    value_ = val;
    SendStat(val);
  }

  /*!
   * \brief Adjust counter value by amount given
   * \param value_change Value to change the counter by
   */
  inline uint64_t IncrementValue(uint64_t value_change) {
    VTUNE_ONLY_CODE(*vtune_ += value_change);
    const uint64_t v = (value_ += value_change);
    SendStat(v);
    return v;
  }

  inline uint64_t DecrementValue(uint64_t value_change) {
    VTUNE_ONLY_CODE(*vtune_ -= value_change);
    const uint64_t v = (value_ -= value_change);
    SendStat(v);
    return v;
  }

  /*! \brief Name of the counter */
  profile_stat_string name_;
  /*! \brief Domain of the counter */
  ProfileDomain *domain_;
  /*! \brief Value of the counter */
  std::atomic<uint64_t>  value_;
  /*! \brief VTune counter object */
  VTUNE_ONLY_CODE(std::unique_ptr<vtune::VTuneCounter> vtune_);
};

class ProfileDuration : public ProfileObject {
 public:
  virtual void start() = 0;
  virtual void stop() = 0;

 protected:
  /*!
   * \brief Basic duration statistic (start time, stop time)
   */
  struct DurationStat : public ProfileStat {
    enum DurationStatIndex {
      kStart, kStop
    };
    /*!
     * \brief Constructor
     * \param begin_event Event type for start point (default is kDurationBegin)
     * \param end_event Event type for stop point (default is kDurationEnd)
     */
    DurationStat(ProfileStat::EventType begin_event = ProfileStat::kDurationBegin,
                 ProfileStat::EventType end_event = ProfileStat::kDurationEnd) {
      items_[kStart].enabled_ = items_[kStop].enabled_ = true;
      items_[kStart].event_type_ = begin_event;
      items_[kStop].event_type_ = end_event;
    }

    /*!
     * \brief Save aggregate data for this stat
     * \param data Stat data
     */
    void SaveAggregate(AggregateStats::StatData *data) const override {
      if (data) {
        data->type_ = AggregateStats::StatData::kDuration;
        ++data->total_count_;
        CHECK_GE(items_[kStop].timestamp_, items_[kStart].timestamp_);
        const uint64_t duration = items_[kStop].timestamp_ - items_[kStart].timestamp_;
        data->total_aggregate_ += duration;
        if (duration > data->max_aggregate_) {
          data->max_aggregate_ = duration;
        }
        if (duration < data->min_aggregate_) {
          data->min_aggregate_ = duration;
        }
      }
    }
  };
};

/*!
 * \brief Task - Thread-granular nestable time block
 */
struct ProfileTask : public ProfileDuration {
  /*!
   * \brief Constructor
   * \param name Name of the task
   * \param domain Domain of the task
   */
  ProfileTask(const char *name, ProfileDomain *domain)
    : name_(name)
      , domain_(domain) {
    CHECK_NOTNULL(domain);
    categories_.set(domain_->name());
    categories_.append(",task");
    VTUNE_ONLY_CODE(vtune_task_.reset(new vtune::VTuneTask(name, domain->dom())));
    NVTX_ONLY_CODE(nvtx_duration_.reset(new common::cuda::NVTXDuration(name)));
  }

  /*!
   * \brief Set the domain
   */
  void setDomain(ProfileDomain* domain) {
    domain_ = domain;
  }

  /*!
   * \brief Start the profiling scope
   */
  void start() override {
    start_time_ = ProfileStat::NowInMicrosec();
    VTUNE_ONLY_CODE(vtune_task_->start());
    NVTX_ONLY_CODE(nvtx_duration_->start());
  }

  /*!
   * \brief Stop the profiling scope
   */
  void stop() override {
    VTUNE_ONLY_CODE(vtune_task_->stop());
    NVTX_ONLY_CODE(nvtx_duration_->stop());
    SendStat();
  }

  ProfileObjectType type() const override { return kTask; }

  /*!
   * \brief Whether to add stat to AggregateStats
   */
  void enableAggregateStats(bool enabled = true) {
    enable_aggregate_ = enabled;
  }

 protected:
  /*!
   * \brief Task statistic object
   */
  struct ProfileTaskStat : public DurationStat {
    explicit ProfileTaskStat(const char *name, uint64_t start_time, uint64_t stop_time)
      : DurationStat(ProfileStat::kAsyncNestableStart, ProfileStat::kAsyncNestableEnd) {
      name_.set(name);
      items_[0].timestamp_ = start_time;
      items_[1].timestamp_ = stop_time;
    }
    void EmitExtra(std::ostream *os, size_t idx) override {
      DurationStat::EmitExtra(os, idx);
      *os << "        \"id\": " << std::hash<std::thread::id>{}(thread_id_) << ",\n";
    }
  };

 private:
  /*!
   * \brief Send this object's statistical datapoint to the profiler
   */
  inline void SendStat() {
    Profiler::Get()->AddNewProfileStat<ProfileTaskStat>([this](ProfileTaskStat *stat) {
      stat->categories_.set(domain_->name());
      stat->enable_aggregate_ = enable_aggregate_;
    }, name_.c_str(), start_time_, ProfileStat::NowInMicrosec());
  }
  /*! \brief Task name */
  const profile_stat_string  name_;
  /*! \brief Task categories */
  profile_stat_string categories_;
  /*! \brief domain */
  ProfileDomain *domain_;
  /*! \brief VTune task object */
  VTUNE_ONLY_CODE(std::unique_ptr<vtune::VTuneTask> vtune_task_);
  /*! \brief NVTX duration object */
  NVTX_ONLY_CODE(std::unique_ptr<common::cuda::NVTXDuration> nvtx_duration_);
  /*! \brief whether to add this stat to AggregateStats */
  bool enable_aggregate_ = true;

 protected:
  /*! \brief Task's start tick */
  uint64_t start_time_;
};

/*!
 * \brief Event - Thread-granular time block
 */
struct ProfileEvent  : public ProfileDuration {
  /*!
   * \brief Constructor
   * \param name Name of the event
   */
  explicit inline ProfileEvent(const char *name)
    : name_(name)
      , categories_("event") {
    VTUNE_ONLY_CODE(vtune_event_ = vtune::VTuneEvent::registry_.get(name));
    NVTX_ONLY_CODE(nvtx_duration_.reset(new common::cuda::NVTXDuration(name)));
  }

  /*!
   * \brief Start the profiling scope
   */
  void start() override {
    start_time_ = ProfileStat::NowInMicrosec();
    VTUNE_ONLY_CODE(vtune_event_->start());
    NVTX_ONLY_CODE(nvtx_duration_->start());
  }

  /*!
   * \brief Stop the profiling scope
   */
  void stop() override {
    VTUNE_ONLY_CODE(vtune_event_->stop());
    SendStat();
  }

  /*!
   * \brief Set catagories (used for chrome tracing)
   * \param categories Comma-delimited categories
   */
  void SetCategories(const char *categories) {
    categories_.set(categories);
  }

  ProfileObjectType type() const override { return kEvent; }

 protected:
  /*!
   * \brief Event statistic object
   */
  struct ProfileEventStat : public DurationStat {
    explicit ProfileEventStat(const char *name, uint64_t start_time, uint64_t stop_time)
      : DurationStat(ProfileStat::kDurationBegin, ProfileStat::kDurationEnd) {
      name_.set(name);
      items_[0].timestamp_ = start_time;
      items_[1].timestamp_ = stop_time;
    }
  };

 private:
  /*!
   * \brief Send this object's statistical datapoint to the profiler
   */
  virtual void SendStat() {
    Profiler::Get()->AddNewProfileStat<ProfileEventStat>([this](ProfileEventStat *stat) {
      stat->categories_.set(categories_.c_str());
    }, name_.c_str(), start_time_, ProfileStat::NowInMicrosec());
  }
  /*! \brief Event name */
  const profile_stat_string  name_;
  /*! \brief Event categories (comma-delimited) */
  profile_stat_string categories_;
  /*! \brief VTune event object */
  VTUNE_ONLY_CODE(vtune::VTuneEvent *vtune_event_);
  /*! \brief NVTX duration object */
  NVTX_ONLY_CODE(std::unique_ptr<common::cuda::NVTXDuration> nvtx_duration_;);

 protected:
  /*! \brief Start time of the event */
  uint64_t start_time_;
};

/*!
 * \brief Frame - Process-granular time block
 */
struct ProfileFrame : public ProfileDuration {
  /*!
   * \brief Constructor
   * \param name Name of the frame
   * \param domain Domain of the frame
   */
  ProfileFrame(const char *name, ProfileDomain *domain)
    : name_(name)
      , domain_(domain) {
    CHECK_NOTNULL(domain);
    categories_.set(domain_->name());
    categories_.append(",frame");
    NVTX_ONLY_CODE(nvtx_duration_.reset(new common::cuda::NVTXDuration(name)));
    VTUNE_ONLY_CODE(vtune_frame_.reset(new vtune::VTuneFrame(domain->dom())));
  }

  /*!
   * \brief Start the profiling scope
   */
  void start() override {
    start_time_ = ProfileStat::NowInMicrosec();
    VTUNE_ONLY_CODE(vtune_frame_->start());
    NVTX_ONLY_CODE(nvtx_duration_->start());
  }

  /*!
   * \brief Stop the profiling scope
   */
  void stop() override {
    VTUNE_ONLY_CODE(vtune_frame_->stop());
    SendStat();
  }

  ProfileObjectType type() const override { return kFrame; }

 protected:
  /*!
   * \brief Frame statistic object
   */
  struct ProfileFrameStat : public DurationStat {
    explicit ProfileFrameStat(const char *name, uint64_t start_time, uint64_t stop_time)
      : DurationStat(ProfileStat::kContextEnter, ProfileStat::kContextLeave) {
      name_.set(name);
      items_[0].timestamp_ = start_time;
      items_[1].timestamp_ = stop_time;
    }
  };

 private:
  /*!
   * \brief Send this object's statistical datapoint to the profiler
   */
  inline void SendStat() {
    Profiler::Get()->AddNewProfileStat<ProfileFrameStat>([this](ProfileFrameStat *stat) {
      stat->categories_.set(categories_.c_str());
    }, name_.c_str(), start_time_, ProfileStat::NowInMicrosec());
  }
  /*! \brief Frame name */
  const profile_stat_string  name_;
  /*! \brief Frame categories (comma-delimited) */
  profile_stat_string categories_;
  /*! \brief Pointer to the domain */
  ProfileDomain *domain_;
  /*! \brief VTune Frame object */
  VTUNE_ONLY_CODE(std::unique_ptr<vtune::VTuneFrame> vtune_frame_);
  /*! \brief NVTX duration object */
  NVTX_ONLY_CODE(std::unique_ptr<common::cuda::NVTXDuration> nvtx_duration_);

 protected:
  /*! \brief Frame start time */
  uint64_t start_time_;
};

/*!
 * \brief Marker - Mark an instance in time
 */
struct ProfileMarker {
  enum MarkerScope {  // Should equal VTune values
    kUnknown, kGlobal, kProcess, kThread, kTask, kMarker
  };

  /*!
   * \brief Constructor
   * \param name Name of the instant marker
   * \param domain Domain of the instant marker
   * \param scope Scope of the instant marker
   * \param nestable true if the instant marker is nestable
   */
  ProfileMarker(const char *name,
                       ProfileDomain *domain,
                       const MarkerScope scope,
                       bool nestable = true)
    : name_(name)
      , domain_(domain)
      , scope_(scope)
      , nestable_(nestable) {
    categories_.set(domain_->name());
    categories_.append(",instant_marker");
    VTUNE_ONLY_CODE(vtune_instant_marker_.reset(
      new vtune::VTuneInstantMarker(name, domain->dom(), static_cast<__itt_scope>(scope))));
  }

  /*!
   * \brief Signal a marker at this instant
   */
  void mark() {
    VTUNE_ONLY_CODE(vtune_instant_marker_->signal());
    SendStat();
  }

 protected:
  /*!
   * \brief Instant-marker statistic object
   */
  struct ProfileMarkerStat : public ProfileStat {
    explicit ProfileMarkerStat(const char *name, const char scope_char, bool nestable)
    : scope_char_(scope_char) {
      items_[0].enabled_ = true;
      items_[0].event_type_ = nestable ? kAsyncNestableInstant : kInstant;
      items_->timestamp_ = NowInMicrosec();
      name_.set(name);
    }
    virtual void EmitExtra(std::ostream *os, size_t idx) {
      ProfileStat::EmitExtra(os, idx);
      *os << "        \"s\": \"" << scope_char_ << "\",\n";
    }
    const char scope_char_;
  };

 private:
  /*!
   * \brief Send this object's statistical datapoint to the profiler
   */
  virtual void SendStat() {
    Profiler::Get()->AddNewProfileStat<ProfileMarkerStat>([this](ProfileMarkerStat *stat) {
      stat->categories_.set(categories_.c_str());
    }, name_.c_str(), vtune_scope_to_chrome_scope(scope_), nestable_);
  }

  static char vtune_scope_to_chrome_scope(const MarkerScope scope) {
    switch (scope) {
      case kThread:
        return 't';
      case kGlobal:
        return 'g';
      case kProcess:
      case kUnknown:
      case kTask:
      case kMarker:
      default:
        return 'p';
    }
  }

  /*! \brief Name of the instant marker */
  const profile_stat_string name_;
  /*! \brief Categories of the instant marker (comma-delimited) */
  profile_stat_string categories_;
  /*! \brief Pointer to the domain of this instant marker */
  ProfileDomain *domain_;
  /*! \brief VTune scope */
  const MarkerScope scope_;
  /*! \brief Whether this marker is nestabe */
  const bool nestable_;
  /*! \brief VTune instant marker object */
  VTUNE_ONLY_CODE(std::unique_ptr<vtune::VTuneInstantMarker> vtune_instant_marker_);
};

static ProfileDomain custom_op_domain("Custom Operator");

/*!
 * \brief Operator profiler object. Logs as both an independent event and a task in
 * the operator domain
 */
struct ProfileOperator : public ProfileEvent {
  /*!
   * \brief Operator attributes
   */
  struct Attributes {
    std::vector<mxnet::TShape> inputs_;
    std::vector<mxnet::TShape> outputs_;
    std::unordered_map<std::string, std::string> attr_;
    std::string to_string() const {
      std::stringstream ss;
      if (!inputs_.empty()) {
        ss << "in: [";
        for (size_t i = 0, n = inputs_.size(); i < n; ++i) {
          if (i) {
            ss << ",";
          }
          ss << inputs_[i];
        }
        ss << "]";
      }
      if (!outputs_.empty()) {
        ss << "out: [";
        for (size_t i = 0, n = outputs_.size(); i < n; ++i) {
          if (i) {
            ss << ",";
          }
          ss << outputs_[i];
        }
        ss << "]";
      }
      if (!attr_.empty()) {
        for (const auto &tt : attr_) {
          ss << " (" << tt.first << "=" << tt.second << ")";
        }
      }
      return ss.str();
    }
  };

  /*!
   * \brief Constructor
   * \param name Name of the operator
   */
  explicit inline ProfileOperator(const char *name, Attributes *attributes)
    : ProfileEvent(name)
      , as_task_(name, &domain_)
      , name_(name)
      , attributes_(attributes)
      , profiling_(!IsDeprecatedOperator(name)) {
    if (IsSubOperatorOfCustom(name)) {
      as_task_.setDomain(&custom_op_domain);
      SetCategories(custom_op_domain.name());
    } else {
      SetCategories(domain_.name());
    }
    // make as_task_ not to add stat to AggregateStats; otherwise we will add twice
    as_task_.enableAggregateStats(false);
  }
  /*!
   * \brief Start the profiling scope
   * \param dev_type Device type that the profiling will occur on
   * \param dev_id Device id associated with this opr
   */
  void startForDevice(mxnet::Context::DeviceType dev_type, uint32_t dev_id) {
    dev_type_ = dev_type;
    dev_id_ = dev_id;
    if (profiling_) {
      ProfileEvent::start();
      as_task_.start();
    }
  }
  /*!
   * \brief Stop the profiling scope
   */
  void stop() override {
    if (profiling_) {
      as_task_.stop();
      ProfileEvent::stop();
    }
  }

  /*!
   * \brief Operation execution statistics
   */
  struct OprExecStat : public DurationStat {
    /*!
     * \brief Constructor
     * \param name Name of the operator
     * \param dev_type Device type (i.e. CPU: 1, GPU: 2, CPUPinned: 3)
     * \param dev_id Device ID (ie GPU number)
     * \param start_time Time when operator starts
     * \param stop_time Time when operator completes
     */
    inline OprExecStat(const char *name, mxnet::Context::DeviceType dev_type, uint32_t dev_id,
                       uint64_t start_time, uint64_t stop_time,
                       const Attributes *attributes)
      : DurationStat(ProfileStat::kDurationBegin, ProfileStat::kDurationEnd)
        , dev_type_(dev_type)
        , dev_id_(dev_id) {
      name_.set(name);
      if (attributes) {
        name_.append(attributes->to_string().c_str());
      }
      if (IsSubOperatorOfCustom(name)) {
        categories_.set(custom_op_domain.name());
      } else {
        categories_.set("operator");
      }
      items_[kStart].timestamp_ = start_time;
      items_[kStop].timestamp_ = stop_time;
    }
    /*! \brief device type: CPU: 1, GPU: 2, CPUPinned: 3 */
    mxnet::Context::DeviceType dev_type_;
    /*! \brief device id */
    uint32_t dev_id_;
  };

 private:
  /*!
   * \brief Send this object's statistical datapoint to the profiler
   */
  void SendStat() override {
    Profiler::Get()->AddNewProfileStat<OprExecStat>(
      [](OprExecStat *stat) {}, name_.c_str(), dev_type_, dev_id_,
      start_time_, ProfileStat::NowInMicrosec(),
      attributes_.get());
  }
  /*!
   * \brief Check if this operator is no longer profiled
   * Notice that this operator may still be used for e.g synchronization
   */
  inline static bool IsDeprecatedOperator(const char* name) {
    return strcmp(name, "CustomOperatorWait") == 0 ||
           strcmp(name, "Custom") == 0 || strcmp(name, "_backward_Custom") == 0;
  }
  /*!
   * \brief Check if this operator a sub-operator of a custom operator
   */
  inline static bool IsSubOperatorOfCustom(const char* name) {
    return strstr(name, "::");
  }
  /*! \brief Also log the operator as a task in the operator domain */
  ProfileTask as_task_;
  /* !\brief Operator name */
  profile_stat_string name_;
  /*! \brief device type: CPU: 1, GPU: 2, CPUPinned: 3 */
  Context::DeviceType dev_type_;
  /*! \brief device id */
  uint32_t dev_id_;
  /*! \brief Operator domain */
  static ProfileDomain domain_;
  /*! \brief Optional operator attributes */
  std::unique_ptr<Attributes> attributes_;
  /*! \brief Whether to profile or not */
  const bool profiling_;
};

/*
 * Profiler inline functions
 */
inline const char *Profiler::DeviceName(mxnet::Context::DeviceType dev_type, int32_t dev_id) {
  return profile_stat[DeviceIndex(dev_type, dev_id)].dev_name_.c_str();
}

inline const char *Profiler::DeviceName(const size_t index) {
  return profile_stat[index].dev_name_.c_str();
}

inline size_t Profiler::DeviceIndex(mxnet::Context::DeviceType dev_type, int32_t dev_id) {
  switch (dev_type) {
    case Context::kCPU:
      return dev_id;
    case Context::kGPU:
      return cpu_num_ + dev_id;
    case Context::kCPUPinned:
      return cpu_num_ + gpu_num_;
    case Context::kCPUShared:
      return cpu_num_ + gpu_num_ + 1;
    default:
      LOG(FATAL) << "Unknown dev_type: " << dev_type;
      return 0;
  }
}

/*!
 * \brief Explicit 'Profiler::AddProfileStat' override for 'OprExecStat'
 * \param opr_stat Unique pointer to the operator statistic
 */
template<>
inline void Profiler::AddProfileStat<ProfileOperator::OprExecStat>(
  std::unique_ptr<ProfileOperator::OprExecStat> *opr_stat) {
  const size_t idx = DeviceIndex((*opr_stat)->dev_type_, (*opr_stat)->dev_id_);
  CHECK_LT(idx, DeviceCount());
  DeviceStats& dev_stat = profile_stat[idx];
  dev_stat.opr_exec_stats_->enqueue((*opr_stat).release());
}

#undef VTUNE_ONLY_CODE  // This macro not meant to be used outside of this file

class ProfilerScope {
 public:
  /*! \brief Get the profiler scope instance */
  static ProfilerScope* Get();
  /*! \brief Set the current profiler scope */
  void SetCurrentProfilerScope(const std::string& scope);
  /*! \brief Get the current profiler scope */
  std::string GetCurrentProfilerScope() const;
 private:
  std::string current_profiler_scope_ = MXNET_STORAGE_DEFAULT_PROFILER_SCOPE_CSTR;
};

}  // namespace profiler
}  // namespace mxnet
#endif  // MXNET_PROFILER_PROFILER_H_
