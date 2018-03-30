/*!
* Copyright by 2018 Contributors
* \file thread_pool.h
* \brief a simple thread pool implementation
* \author Philip Cho
*/
#ifndef TREELITE_THREAD_POOL_THREAD_POOL_H_
#define TREELITE_THREAD_POOL_THREAD_POOL_H_

#include "spsc_queue.h"
#include <treelite/common.h>
#include <sched.h>

namespace treelite {

template <typename InputToken, typename OutputToken, typename TaskContext>
class ThreadPool {
 public:
  using TaskFunc = void(*)(SpscQueue<InputToken>*, SpscQueue<OutputToken>*,
                           const TaskContext*);

  ThreadPool(int num_worker, const TaskContext* context, TaskFunc task)
    : num_worker_(num_worker), context_(context), task_(task) {
    CHECK(num_worker_ > 0
          && num_worker_ + 1 <= std::thread::hardware_concurrency())
    << "Number of worker threads must be between 1 and "
    << std::thread::hardware_concurrency() - 1;
    LOG(INFO) << "new thread pool with " << num_worker_ << " worker threads";
    for (int i = 0; i < num_worker_; ++i) {
      incoming_queue_.emplace_back(common::make_unique<SpscQueue<InputToken>>());
      outgoing_queue_.emplace_back(common::make_unique<SpscQueue<OutputToken>>());
    }
    thread_.resize(num_worker_);
    for (int i = 0; i < num_worker_; ++i) {
      thread_[i] = std::thread(task_, incoming_queue_[i].get(),
                                      outgoing_queue_[i].get(),
                                      context_);
    }
    /* bind threads to cores */
    SetAffinity();
  }
  ~ThreadPool() {
    LOG(INFO) << "delete thread pool";
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
  TaskFunc task_;
  const TaskContext* context_;

  inline void SetAffinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    for (int i = 0; i < num_worker_; ++i) {
      const int core_id = i + 1;
      CPU_ZERO(&cpuset);
      CPU_SET(core_id, &cpuset);
      pthread_setaffinity_np(thread_[i].native_handle(),
                             sizeof(cpu_set_t), &cpuset);
    }
  }
};

}  // namespace treelite

#endif  // TREELITE_THREAD_POOL_THREAD_POOL_H_
