/*!
* Copyright (c) 2018-2021 by Contributors
* \file spsc_queue.h
* \brief Lock-free single-producer-single-consumer queue
* \author Yida Wang, Hyunsu Cho
*/
#ifndef TREELITE_PREDICTOR_THREAD_POOL_SPSC_QUEUE_H_
#define TREELITE_PREDICTOR_THREAD_POOL_SPSC_QUEUE_H_

#include <treelite/logging.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdint>

namespace treelite {
namespace predictor {

const constexpr int kL1CacheBytes = 64;

/*! \brief Lock-free single-producer-single-consumer queue for each thread */
template <typename T>
class SpscQueue {
 public:
  SpscQueue() :
    buffer_(new T[kRingSize]),
    head_(0),
    tail_(0) {
  }

  ~SpscQueue() {
    delete[] buffer_;
  }

  void Push(const T& input) {
    while (!Enqueue(input)) {
      std::this_thread::yield();
    }
    if (pending_.fetch_add(1) == -1) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.notify_one();
    }
  }

  bool Pop(T* output, std::uint32_t spin_count = 300000) {
    // Busy wait a bit when the queue is empty.
    // If a new element comes to the queue quickly, this wait avoid the worker
    // from sleeping.
    // The default spin count is set by following the typical omp convention
    for (std::uint32_t i = 0; i < spin_count && pending_.load() == 0; ++i) {
      std::this_thread::yield();
    }
    if (pending_.fetch_sub(1) == 0) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
        return pending_.load() >= 0 || exit_now_.load();
      });
    }
    if (exit_now_.load(std::memory_order_relaxed)) {
      return false;
    }
    const std::uint32_t head = head_.load(std::memory_order_relaxed);
    // sanity check if the queue is empty
    TREELITE_CHECK(tail_.load(std::memory_order_acquire) != head);
    *output = buffer_[head];
    head_.store((head + 1) % kRingSize, std::memory_order_release);
    return true;
  }

  /*!
   * \brief Signal to terminate the worker.
   */
  void SignalForKill() {
    std::lock_guard<std::mutex> lock(mutex_);
    exit_now_.store(true);
    cv_.notify_all();
  }

 protected:
  bool Enqueue(const T& input) {
    const std::uint32_t tail = tail_.load(std::memory_order_relaxed);

    if ((tail + 1) % kRingSize != (head_.load(std::memory_order_acquire))) {
      buffer_[tail] = input;
      tail_.store((tail + 1) % kRingSize, std::memory_order_release);
      return true;
    }
    return false;
  }

  // the cache line paddings are used for avoid false sharing between atomic variables
  typedef char cache_line_pad_t[kL1CacheBytes];
  cache_line_pad_t pad0_;
  // size of the queue, the queue can host size_ - 1 items at most
  // define it as a constant for better compiler optimization
  static constexpr const int kRingSize = 2;
  // pointer to access the item
  T* const buffer_;

  cache_line_pad_t pad1_;
  // queue head, where one gets an element from the queue
  std::atomic<std::uint32_t> head_;

  cache_line_pad_t pad2_;
  // queue tail, when one puts an element to the queue
  std::atomic<std::uint32_t> tail_;

  cache_line_pad_t pad3_;
  // pending elements in the queue
  std::atomic<std::int8_t> pending_{0};

  cache_line_pad_t pad4_;
  // signal for exit now
  std::atomic<bool> exit_now_{false};

  // internal mutex
  std::mutex mutex_;
  // cv for consumer
  std::condition_variable cv_;
};

}  // namespace predictor
}  // namespace treelite

#endif  // TREELITE_PREDICTOR_THREAD_POOL_SPSC_QUEUE_H_
