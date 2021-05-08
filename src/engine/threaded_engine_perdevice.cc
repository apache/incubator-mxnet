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
 * \file threaded_engine_perdevice.cc
 * \brief ThreadedEngine that uses fix amount of thread for each device.
 */
#include <dmlc/base.h>
#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/concurrency.h>
#include <dmlc/thread_group.h>

#include <memory>
#include "../initialize.h"
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "../common/lazy_alloc_array.h"
#include "../common/utils.h"

namespace mxnet {
namespace engine {
/*!
 * \brief ThreadedEngine uses per device threads.
 * The policy of this Engine:
 *  - Execute Async operation immediately if pushed from Pusher.
 *  - Use fixed amount of threads for each device.
 *  - Use special threads for copy operations.
 *  - Each stream is allocated and bound to each of the thread.
 */
class ThreadedEnginePerDevice : public ThreadedEngine {
 public:
  static auto constexpr kFIFO = dmlc::ConcurrentQueueType::kFIFO;
  static auto constexpr kPriority = dmlc::ConcurrentQueueType::kPriority;
  static auto constexpr kCopyQueue = kPriority;
  static auto constexpr kPriorityQueue = kPriority;
  static auto constexpr kWorkerQueue = kFIFO;

  ThreadedEnginePerDevice() noexcept(false) {
    this->Start();
  }
  ~ThreadedEnginePerDevice() noexcept(false) override {
    this->StopNoWait();
  }

  void StopNoWait() {
    SignalQueuesForKill();
    gpu_normal_workers_.Clear();
    gpu_priority_workers_.Clear();
    gpu_copy_workers_.Clear();
    cpu_normal_workers_.Clear();
    cpu_priority_worker_.reset(nullptr);
  }

  void Stop() override {
    if (is_worker_) return;
    WaitForAll();
    StopNoWait();
  }

  void Start() override {
    if (is_worker_) return;
    gpu_worker_nthreads_ = common::GetNumThreadsPerGPU();
    // MXNET_CPU_WORKER_NTHREADS
    cpu_worker_nthreads_ = LibraryInitializer::Get()->cpu_worker_nthreads_;
    gpu_copy_nthreads_ = dmlc::GetEnv("MXNET_GPU_COPY_NTHREADS", 2);
    // create CPU task
    int cpu_priority_nthreads = dmlc::GetEnv("MXNET_CPU_PRIORITY_NTHREADS", 4);
    cpu_priority_worker_ = std::make_unique<ThreadWorkerBlock<kPriorityQueue>>();
    cpu_priority_worker_->pool = std::make_unique<ThreadPool>(
        cpu_priority_nthreads,
        [this](std::shared_ptr<dmlc::ManualEvent> ready_event) {
          this->CPUWorker(Context(), cpu_priority_worker_.get(), ready_event);
        }, true);
    // GPU tasks will be created lazily
  }

 protected:
  void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
    const Context& ctx = opr_block->ctx;
    if ((opr_block->opr->prop == FnProperty::kAsync ||
         opr_block->opr->prop == FnProperty::kDeleteVar) && pusher_thread) {
      if (ctx.dev_mask() == Context::kGPU) {
        #if MXNET_USE_CUDA
        MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(ctx.dev_id));
        #endif
      }
      this->ExecuteOprBlock(RunContext{ctx, nullptr, nullptr, false}, opr_block);
    } else {
      if (ctx.dev_mask() == Context::kCPU) {
        // CPU execution.
        if (opr_block->opr->prop == FnProperty::kCPUPrioritized) {
          cpu_priority_worker_->task_queue.Push(opr_block, opr_block->priority);
        } else {
          int dev_id = ctx.dev_id;
          int nthread = cpu_worker_nthreads_;
          auto ptr =
          cpu_normal_workers_.Get(dev_id, [this, ctx, nthread]() {
              auto blk = new ThreadWorkerBlock<kWorkerQueue>();
              blk->pool = std::make_unique<ThreadPool>(nthread,
                  [this, ctx, blk](std::shared_ptr<dmlc::ManualEvent> ready_event) {
                    this->CPUWorker(ctx, blk, ready_event);
                  }, true);
            return blk;
          });
          if (ptr) {
            if (opr_block->opr->prop == FnProperty::kDeleteVar) {
              ptr->task_queue.PushFront(opr_block, opr_block->priority);
            } else {
              ptr->task_queue.Push(opr_block, opr_block->priority);
            }
          }
        }
      } else {
        CHECK_EQ(ctx.dev_mask(), Context::kGPU);
        // GPU execution.
        const FnProperty prop = opr_block->opr->prop;
        const bool is_copy = (prop == FnProperty::kCopyFromGPU ||
                              prop == FnProperty::kCopyToGPU);
        if (is_copy) {
          const size_t nthread = gpu_copy_nthreads_;
          auto ptr = gpu_copy_workers_.Get(ctx.dev_id, [this, ctx, is_copy, nthread]() {
            // Signify to kernel that GPU is being used, so reserve cores as necessary
            OpenMP::Get()->set_reserve_cores(GetReserveCoreCount(true));
            auto blk = new ThreadWorkerBlock<kCopyQueue>();
              blk->pool = std::make_unique<ThreadPool>(
                nthread,
                [this, ctx, is_copy, blk]
                  (std::shared_ptr<dmlc::ManualEvent> ready_event) {
                    this->GPUWorker(ctx, is_copy, blk, ready_event);
                  }, true);
              return blk;
            });
          if (ptr) {
            if (opr_block->opr->prop == FnProperty::kDeleteVar) {
              ptr->task_queue.PushFront(opr_block, opr_block->priority);
            } else {
              ptr->task_queue.Push(opr_block, opr_block->priority);
            }
          }
        } else {
          const size_t nthread = gpu_worker_nthreads_;
          // GPU priority task
          if (opr_block->opr->prop == FnProperty::kGPUPrioritized) {
            auto ptr = gpu_priority_workers_.Get(ctx.dev_id, [this, ctx, is_copy, nthread]() {
              // Signify to kernel that GPU is being used, so reserve cores as necessary
              OpenMP::Get()->set_reserve_cores(GetReserveCoreCount(true));
                auto blk = new ThreadWorkerBlock<kPriorityQueue>();
                blk->pool = std::make_unique<ThreadPool>(
                  nthread,
                  [this, ctx, is_copy, blk]
                    (std::shared_ptr<dmlc::ManualEvent> ready_event) {
                      this->GPUWorker(ctx, is_copy, blk, ready_event);
                    }, true);
                return blk;
            });
            if (ptr) {
              ptr->task_queue.Push(opr_block, opr_block->priority);
            }
          } else {
            // GPU normal task
            auto ptr = gpu_normal_workers_.Get(ctx.dev_id, [this, ctx, is_copy, nthread]() {
              // Signify to kernel that GPU is being used, so reserve cores as necessary
              OpenMP::Get()->set_reserve_cores(GetReserveCoreCount(true));
                auto blk = new ThreadWorkerBlock<kWorkerQueue>();
                blk->pool = std::make_unique<ThreadPool>(
                  nthread,
                  [this, ctx, is_copy, blk]
                    (std::shared_ptr<dmlc::ManualEvent> ready_event) {
                      this->GPUWorker(ctx, is_copy, blk, ready_event);
                    }, true);
                return blk;
            });
            if (ptr) {
              if (opr_block->opr->prop == FnProperty::kDeleteVar) {
                ptr->task_queue.PushFront(opr_block, opr_block->priority);
              } else {
                ptr->task_queue.Push(opr_block, opr_block->priority);
              }
            }
          }
        }
      }
    }
  }

 private:
  // working unit for each of the task.
  template<dmlc::ConcurrentQueueType type>
  struct ThreadWorkerBlock {
    // task queue on this task
    dmlc::ConcurrentBlockingQueue<OprBlock*, type>  task_queue;
    // thread pool that works on this task
    std::unique_ptr<ThreadPool> pool;
    // constructor
    ThreadWorkerBlock() = default;
    // destructor
    ~ThreadWorkerBlock() = default;
  };

  /*! \brief whether this is a worker thread. */
  static MX_THREAD_LOCAL bool is_worker_;
  /*! \brief number of concurrent thread cpu worker uses */
  size_t cpu_worker_nthreads_;
  /*! \brief number of concurrent thread each gpu worker uses */
  size_t gpu_worker_nthreads_;
  /*! \brief number of concurrent thread each gpu copy worker uses */
  size_t gpu_copy_nthreads_;
  // cpu worker
  common::LazyAllocArray<ThreadWorkerBlock<kWorkerQueue> > cpu_normal_workers_;
  // cpu priority worker
  std::unique_ptr<ThreadWorkerBlock<kPriorityQueue> > cpu_priority_worker_;
  // workers doing normal works on GPU
  common::LazyAllocArray<ThreadWorkerBlock<kWorkerQueue> > gpu_normal_workers_;
  // workers doing copy works from/to GPU
  common::LazyAllocArray<ThreadWorkerBlock<kCopyQueue> > gpu_copy_workers_;
  // gpu priority workers
  common::LazyAllocArray<ThreadWorkerBlock<kPriorityQueue> > gpu_priority_workers_;
  /*!
   * \brief GPU worker that performs operations on a certain device.
   * \param dev_id The device id of the worker.
   * \param is_copy_worker whether the worker only do copy job
   * \param block The task block of the worker.
   */
  template<dmlc::ConcurrentQueueType type>
  inline void GPUWorker(Context ctx,
                        bool is_copy_worker,
                        ThreadWorkerBlock<type> *block,
                        const std::shared_ptr<dmlc::ManualEvent>& ready_event) {
    this->is_worker_ = true;
#if MXNET_USE_CUDA
    CHECK(block != nullptr);
    mshadow::Stream<gpu> *stream = nullptr;
    GPUAuxStream *aux_stream = nullptr;
    do {
      ThreadPool::SetReadyOnDestroy setReady(ready_event);
      // allocate stream
      mshadow::SetDevice<gpu>(ctx.dev_id);
      if (is_copy_worker) {
        stream = mshadow::NewStream<gpu>(false, false, ctx.dev_id);
      } else {
        stream = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0, ctx.dev_id);
        aux_stream = new GPUAuxStream(stream);
      }
    } while (false);
    // execute task
    OprBlock* opr_block;
    RunContext run_ctx{ctx, stream, aux_stream, false};
    auto* task_queue = &(block->task_queue);

    // Don't eat up omp threads for GPU jobs.  They're probably best used elsewhere,
    // for example for image decoding or the optimizer pass
    OpenMP::Get()->on_start_worker_thread(false);

    while (task_queue->Pop(&opr_block)) {
      this->ExecuteOprBlock(run_ctx, opr_block);
    }
    // Catch exception for CUDA driver shutdown
    MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(stream));
    if (aux_stream != nullptr)
      delete aux_stream;
#else
    ready_event->signal();
#endif
  }
  /*!
   * \brief CPU worker that performs operations on CPU.
   * \param block The task block of the worker.
   */
  template<dmlc::ConcurrentQueueType type>
  inline void CPUWorker(Context ctx,
                        ThreadWorkerBlock<type> *block,
                        const std::shared_ptr<dmlc::ManualEvent>& ready_event) {
    this->is_worker_ = true;
    auto* task_queue = &(block->task_queue);
    RunContext run_ctx{ctx, nullptr, nullptr, false};

    // execute task
    OprBlock* opr_block;
    ready_event->signal();

    // Set default number of threads for OMP parallel regions initiated by this thread
    OpenMP::Get()->on_start_worker_thread(true);

    while (task_queue->Pop(&opr_block)) {
      this->ExecuteOprBlock(run_ctx, opr_block);
    }
  }

  /*!
   * \brief Get number of cores this engine should reserve for its own use
   * \param using_gpu Whether there is GPU usage
   * \return number of cores that this engine wishes to be reserved
   * \note Testing found no degradation of performance using these values
   *       running cifar10 with resnet50 on various GPU systems,
   *       including AWS p2.16xlarge, which has 16 GPU's
   */
  int GetReserveCoreCount(const bool using_gpu) const {
    int reserve = 0;
    if (using_gpu) {
      // Save at least one for GPU tasks
      ++reserve;
      // If we have 8 or more real cores, reserve another core for GPU tasks
      if (OpenMP::Get()->GetRecommendedOMPThreadCount(true) >= 8) {
        ++reserve;
      }
    }
    return reserve;
  }

  /*! \brief Signal a single queue for shutdown */
  template<typename Object>
  static inline void SignalQueueForKill(common::LazyAllocArray<Object> *array) {
    array->ForEach([](size_t i, Object *block) {
      block->task_queue.SignalForKill();
    });
  }

  /*! Signal all queues for shutdown */
  void SignalQueuesForKill() {
    SignalQueueForKill(&gpu_priority_workers_);
    SignalQueueForKill(&gpu_normal_workers_);
    SignalQueueForKill(&gpu_copy_workers_);
    SignalQueueForKill(&cpu_normal_workers_);
    if (cpu_priority_worker_) {
      cpu_priority_worker_->task_queue.SignalForKill();
    }
  }
};

Engine *CreateThreadedEnginePerDevice() {
  return new ThreadedEnginePerDevice();
}

MX_THREAD_LOCAL bool ThreadedEnginePerDevice::is_worker_ = false;

}  // namespace engine
}  // namespace mxnet
