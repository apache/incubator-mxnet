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
 *  Copyright (c) 2015 by Contributors
 * \file naive_engine.cc
 * \brief Implementation of NaiveEngine
 */
#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <vector>
#include "./engine_impl.h"
#include "../profiler/profiler.h"
#include "./openmp.h"
#include "../common/object_pool.h"
#include "../profiler/custom_op_profiler.h"

namespace mxnet {
namespace engine {

/*!
 * \brief var used in Naive Engine for tracking the version
 * of the objects it is associated with.
 */
class NaiveVar final
    : public Var, public common::ObjectPoolAllocatable<NaiveVar> {
 public:
  inline static NaiveVar* CastFromBase(Var* ptr) {
    return ptr->Cast<NaiveVar>();
  }
};  // class NaiveVar


// implement naive engine
class NaiveEngine final : public Engine {
 public:
  struct NaiveOpr : public Opr {
    AsyncFn fn;
    std::vector<VarHandle> const_vars;
    std::vector<VarHandle> mutable_vars;
    FnProperty prop;
    std::string opr_name;
    /*! \brief indicate whether to profile this operator */
    bool profiling{false};
    /*! \brief operator execution statistics */
    std::unique_ptr<profiler::ProfileOperator> opr_profile;
  };

  NaiveEngine() {
    objpool_opr_ref_ = common::ObjectPool<NaiveOpr>::_GetSharedRef();
    objpool_var_ref_ = common::ObjectPool<NaiveVar>::_GetSharedRef();
  }
  // virtual destructor
#if MXNET_USE_CUDA
  ~NaiveEngine() override {
    LOG(INFO) << "Engine shutdown";
    for (size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i] != nullptr) {
        // Catch exception for CUDA driver shutdown
        MSHADOW_CATCH_ERROR(mshadow::DeleteStream(streams_[i]));
        streams_[i] = nullptr;
      }
    }
    for (size_t i = 0; i < aux_streams_.size(); ++i) {
      if (aux_streams_[i] != nullptr) {
        delete aux_streams_[i];
        aux_streams_[i] = nullptr;
      }
    }
  }
#else
  ~NaiveEngine() override = default;
#endif

  void Stop() override {
  }

  void Start() override {
  }

  // new variables
  VarHandle NewVariable() override {
    return NaiveVar::New();
  }

  OprHandle NewOperator(AsyncFn fn,
                        std::vector<VarHandle> const& const_vars,
                        std::vector<VarHandle> const& mutable_vars,
                        FnProperty prop = FnProperty::kNormal,
                        const char* opr_name = nullptr,
                        bool wait = false) override {
    NaiveOpr *opr = new NaiveOpr();
    opr->fn = fn;
    opr->const_vars = const_vars;
    opr->mutable_vars = mutable_vars;
    opr->prop = prop;
    opr->opr_name = opr_name ? std::string(opr_name) : std::string();
    return opr;
  }

  void DeleteOperator(OprHandle op) override {
    NaiveOpr *opr = op->Cast<NaiveOpr>();
    delete opr;
  }

  void Push(OprHandle op, Context exec_ctx, int priority = 0, bool profiling = false) override {
    profiler::Profiler *profiler = profiler::Profiler::Get();
    NaiveOpr *opr = op->Cast<NaiveOpr>();
    opr->profiling = profiling && profiler->IsProfiling(profiler::Profiler::kSymbolic);
    this->PushAsync([&](RunContext ctx, CallbackOnComplete on_complete) {
        if (opr->profiling) {
          std::unique_ptr<profiler::ProfileOperator::Attributes> attrs;
          if (profiler->AggregateEnabled()) {
            attrs = std::make_unique<profiler::ProfileOperator::Attributes>();
          }
          opr->opr_profile = std::make_unique<profiler::ProfileOperator>(opr->opr_name.c_str(),
                                                               attrs.release());
          opr->opr_profile->startForDevice(exec_ctx.dev_type, exec_ctx.dev_id);
        }
        opr->fn(ctx, on_complete);
        if (opr->profiling) {
          opr->opr_profile->stop();
        }
      },
      exec_ctx,
      opr->const_vars,
      opr->mutable_vars,
      opr->prop,
      priority,
      opr->opr_name.c_str());
  }

/*!
 * \brief NaiveEngine's PushAsync was intentionally synchronous.
 * User should not make any assumption about execution order when using async interface of any engine.
 */
  void PushAsync(AsyncFn exec_fun,
                 Context exec_ctx,
                 std::vector<VarHandle> const& const_vars,
                 std::vector<VarHandle> const& mutable_vars,
                 FnProperty prop = FnProperty::kNormal,
                 int priority = 0,
                 const char* opr_name = nullptr,
                 bool wait = false) override {
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    CallbackOnComplete callback = CreateCallback(
        NaiveEngine::OnComplete, &promise);
    profiler::Profiler *profiler = profiler::Profiler::Get();
    auto opr_deleter = [this](NaiveOpr* p) {
      this->DeleteOperator(p);
    };
    std::unique_ptr<NaiveOpr, decltype(opr_deleter)> opr(nullptr, opr_deleter);
    const bool profiling = opr_name && profiler->IsProfiling(profiler::Profiler::kImperative);
    // GenerateDisplayName() will return a pointer to the correct name of the operator
    const char* display_name = profiling ?
                               profiler::CustomOpProfiler::Get()->GenerateDisplayName(opr_name) :
                               opr_name;
    if (profiling) {
      opr.reset(NewOperator(exec_fun, const_vars, mutable_vars,
                        prop, display_name)->Cast<NaiveOpr>());
      opr->profiling = profiling;
      std::unique_ptr<profiler::ProfileOperator::Attributes> attrs;
      if (profiler->AggregateEnabled()) {
        attrs = std::make_unique<profiler::ProfileOperator::Attributes>();
      }
      opr->opr_profile = std::make_unique<profiler::ProfileOperator>(opr->opr_name.c_str(),
                                                                     attrs.release());
      opr->opr_profile->startForDevice(exec_ctx.dev_type, exec_ctx.dev_id);
    }
    if (exec_ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
      size_t dev_id = static_cast<size_t>(exec_ctx.dev_id);
      cudaGetLastError();  // reset cuda error
      MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(exec_ctx.dev_id));
      if (streams_.size() <= dev_id) {
        streams_.resize(dev_id + 1, nullptr);
        aux_streams_.resize(dev_id + 1, nullptr);
      }
      if (streams_[dev_id] == nullptr) {
        streams_[dev_id] = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0, dev_id);
        aux_streams_[dev_id] = new GPUAuxStream(streams_[dev_id]);
      }
      exec_fun(RunContext{exec_ctx, streams_[dev_id], aux_streams_[dev_id], false}, callback);
#else
      LOG(FATAL) << "GPU is not enabled";
#endif
    } else {
      exec_fun(RunContext{exec_ctx, &cpu_stream_, nullptr, false}, callback);
    }
    future.wait();
    // increment mutable var version
    for (auto var : mutable_vars) {
      ++var->version_;
    }
    if (profiling) {
      opr->opr_profile->stop();
    }
  }

  void DeleteVariable(SyncFn delete_fn, Context exec_ctx, VarHandle var) override {
    NaiveVar* naive_var = NaiveVar::CastFromBase(var);
    this->PushAsync([delete_fn, naive_var](RunContext ctx, CallbackOnComplete on_complete) mutable {
        delete_fn(ctx);
        NaiveVar::Delete(naive_var);
        on_complete();
      }, exec_ctx, {}, {var}, FnProperty::kDeleteVar, 0, "DeleteVariable");
  }

  void WaitForVar(VarHandle var) override {
  }

  void WaitForAll() override {
  }

  void Throw(VarHandle var) override {
  }

  void NotifyShutdown() override {
    shutdown_phase_.store(true);
  }

 private:
  // callback to oncomplete
  static void OnComplete(Engine *engine, void *param,
                         const dmlc::Error* error) {
    static_cast<std::promise<void>*>(param)->set_value();
  }
  /*! \brief whether it is during shutdown phase*/
  std::atomic<bool> shutdown_phase_{false};
  // CPU stream
  mshadow::Stream<cpu> cpu_stream_;
  // GPU streams
  std::vector<mshadow::Stream<gpu>*> streams_;
#if MXNET_USE_CUDA
  // GPU auxiliary streams
  std::vector<GPUAuxStream*> aux_streams_;
#endif
/*!
 * \brief Holding a shared_ptr to the object pool to prevent it from being destructed too early
 * See also #309 (https://github.com/dmlc/mxnet/issues/309) and similar fix in threaded_engine.h.
 * Without this, segfaults seen on CentOS7 in test_operator_gpu.py:test_convolution_multiple_streams
 */
  std::shared_ptr<common::ObjectPool<NaiveOpr> > objpool_opr_ref_;
  std::shared_ptr<common::ObjectPool<NaiveVar> > objpool_var_ref_;
};  // class NaiveEngine

Engine *CreateNaiveEngine() {
  return new NaiveEngine();
}

}  // namespace engine
}  // namespace mxnet
