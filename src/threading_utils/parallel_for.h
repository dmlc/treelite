//
// Created by phcho on 7/15/21.
//

#ifndef TREELITE_THREADING_UTILS_PARALLEL_FOR_H_
#define TREELITE_THREADING_UTILS_PARALLEL_FOR_H_

#include <thread>
#include <vector>
#include <cstddef>

namespace {

template <typename IndexType>
std::vector<IndexType> ComputeWorkRange(IndexType begin, IndexType end, std::size_t nthread) {
  IndexType num_elem = end - begin;
  const IndexType portion = num_elem / nthread + !!(num_elem % nthread);
  // integer division, rounded-up

  std::vector<IndexType> work_range(nthread + 1);
  work_range[0] = begin;
  std::size_t acc = begin;
  for (std::size_t i = 0; i < nthread; ++i) {
    acc += portion;
    work_range[i + 1] = std::min(acc, end);
  }
  TREELITE_CHECK_EQ(work_range[nthread], end);

  return work_range;
}

}  // anonymous namespace

namespace treelite {
namespace threading_utils {

template <typename IndexType, typename FuncType>
void ParallelFor(IndexType begin, IndexType end, std::size_t nthread, FuncType func) {
  TREELITE_CHECK_GE(end, begin);
  if (begin == end) {
    return;
  }
  /* Divide the rnage [begin, end) equally among the threads.
   * The i-th thread gets the range [work_range[i], work_range[i+1]). */
  std::vector<IndexType> work_range = ComputeWorkRange(begin, end, nthread);

  // Launch (nthread - 1) threads, as the main thread should also perform work.
  std::vector<std::thread> threads;
  for (std::size_t thread_id = 1; thread_id < nthread; ++thread_id) {
    threads.emplace_back([&work_range, &func, thread_id]() {
      const IndexType begin_ = work_range[thread_id];
      const IndexType end_ = work_range[thread_id + 1];
      for (IndexType i = begin_; i < end_; ++i) {
        func(i, thread_id);
      }
    });
  }
  {
    const IndexType begin_ = work_range[0];
    const IndexType end_ = work_range[1];
    for (IndexType i = begin_; i < end_; ++i) {
      func(i, 0);
    }
  }
  // Join threads
  for (std::thread& thread : threads) {
    thread.join();
  }
}

}  // namespace threading_utils
}  // namespace treelite

#endif  // TREELITE_THREADING_UTILS_PARALLEL_FOR_H_
