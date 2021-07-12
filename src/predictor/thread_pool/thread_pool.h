/*!
* Copyright (c) 2018-2020 by Contributors
* \file thread_pool.h
* \brief a simple thread pool implementation
* \author Hyunsu Cho
*/
#ifndef TREELITE_PREDICTOR_THREAD_POOL_THREAD_POOL_H_
#define TREELITE_PREDICTOR_THREAD_POOL_THREAD_POOL_H_

#include <string>
#include <memory>
#include <vector>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <mach/mach_init.h>
#include <mach/thread_policy.h>
#endif
#include "spsc_queue.h"

namespace treelite {
namespace predictor {

template <typename InputToken, typename OutputToken, typename TaskContext>
class ThreadPool {
 public:
  using TaskFunc = void(*)(SpscQueue<InputToken>*, SpscQueue<OutputToken>*,
                           const TaskContext*);

  ThreadPool(int num_worker, const TaskContext* context, TaskFunc task)
    : num_worker_(num_worker), context_(context), task_(task) {
    TREELITE_CHECK(num_worker_ >= 0
                   && static_cast<unsigned>(num_worker_) < std::thread::hardware_concurrency())
    << "Number of worker threads must be between 0 and "
    << (std::thread::hardware_concurrency() - 1);
    for (int i = 0; i < num_worker_; ++i) {
      incoming_queue_.emplace_back(new SpscQueue<InputToken>());
      outgoing_queue_.emplace_back(new SpscQueue<OutputToken>());
    }
    thread_.resize(num_worker_);
    for (int i = 0; i < num_worker_; ++i) {
      thread_[i] = std::thread(task_, incoming_queue_[i].get(),
                                      outgoing_queue_[i].get(),
                                      context_);
    }
    /* bind threads to cores */
    const char* bind_flag = getenv("TREELITE_BIND_THREADS");
    if (bind_flag == nullptr || std::stoi(bind_flag) == 1) {
      SetAffinity();
    }
  }
  ~ThreadPool() {
    for (int i = 0; i < num_worker_; ++i) {
      incoming_queue_[i]->SignalForKill();
      outgoing_queue_[i]->SignalForKill();
      thread_[i].join();
    }
  }

  void SubmitTask(int tid, InputToken request) {
    incoming_queue_[tid]->Push(request);
  }

  bool WaitForTask(int tid, OutputToken* response) {
    return outgoing_queue_[tid]->Pop(response);
  }

 private:
  int num_worker_;
  std::vector<std::thread> thread_;
  std::vector<std::unique_ptr<SpscQueue<InputToken>>> incoming_queue_;
  std::vector<std::unique_ptr<SpscQueue<OutputToken>>> outgoing_queue_;
  const TaskContext* context_;
  TaskFunc task_;

  inline void SetAffinity() {
#ifdef _WIN32
    /* Windows */
    SetThreadAffinityMask(GetCurrentThread(), 0x1);
    for (int i = 0; i < num_worker_; ++i) {
      const int core_id = i + 1;
      SetThreadAffinityMask(thread_[i].native_handle(), (1ULL << core_id));
    }
#elif defined(__APPLE__) && defined(__MACH__)
#include <TargetConditionals.h>
#if TARGET_OS_MAC == 1
    /* Mac OSX */
    thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
    thread_affinity_policy_data_t policy = {0};
    thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
    for (int i = 0; i < num_worker_; ++i) {
      const int core_id = i + 1;
      mach_thread = pthread_mach_thread_np(thread_[i].native_handle());
      policy = {core_id};
      thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                        (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
    }
#else
    #error "iPhone not supported yet"
#endif
#else
    /* Linux and others */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
#if defined(__ANDROID__)
    sched_setaffinity(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
    for (int i = 0; i < num_worker_; ++i) {
      const int core_id = i + 1;
      CPU_ZERO(&cpuset);
      CPU_SET(core_id, &cpuset);
#if defined(__ANDROID__)
      sched_setaffinity(thread_[i].native_handle(),
                        sizeof(cpu_set_t), &cpuset);
#else
      pthread_setaffinity_np(thread_[i].native_handle(),
                             sizeof(cpu_set_t), &cpuset);
#endif
    }
#endif
  }
};

}  // namespace predictor
}  // namespace treelite

#endif  // TREELITE_PREDICTOR_THREAD_POOL_THREAD_POOL_H_
