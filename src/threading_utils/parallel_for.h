/*!
* Copyright (c) 2021 by Contributors
* \file parallel_for.h
* \brief Implemenation of parallel for loop
* \author Hyunsu Cho
*/
#ifndef TREELITE_THREADING_UTILS_PARALLEL_FOR_H_
#define TREELITE_THREADING_UTILS_PARALLEL_FOR_H_

#include <treelite/omp.h>
#include <treelite/logging.h>
#include <type_traits>
#include <algorithm>
#include <exception>
#include <mutex>
#include <cstddef>
#include <cstdint>

namespace treelite {
namespace threading_utils {

/*!
 * \brief OMP Exception class catches, saves and rethrows exception from OMP blocks
 */
class OMPException {
 private:
  // exception_ptr member to store the exception
  std::exception_ptr omp_exception_;
  // mutex to be acquired during catch to set the exception_ptr
  std::mutex mutex_;

 public:
  /*!
   * \brief Parallel OMP blocks should be placed within Run to save exception
   */
  template <typename Function, typename... Parameters>
  void Run(Function f, Parameters... params) {
    try {
      f(params...);
    } catch (std::exception& ex) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!omp_exception_) {
        omp_exception_ = std::current_exception();
      }
    }
  }

  /*!
   * \brief should be called from the main thread to rethrow the exception
   */
  void Rethrow() {
    if (this->omp_exception_) {
      std::rethrow_exception(this->omp_exception_);
    }
  }
};

inline int OmpGetThreadLimit() {
  int limit = omp_get_thread_limit();
  TREELITE_CHECK_GE(limit, 1) << "Invalid thread limit for OpenMP.";
  return limit;
}

inline int MaxNumThread() {
  return std::min(std::min(omp_get_num_procs(), omp_get_max_threads()), OmpGetThreadLimit());
}

/*!
 * \brief Represent thread configuration, to be used with parallel loops.
 */
struct ThreadConfig {
  std::uint32_t nthread;
};

/*!
 * \brief Create therad configuration.
 * @param nthread Number of threads to use. If \<= 0, use all available threads. This value is
 *                validated to ensure that it's in a valid range.
 * @return Thread configuration
 */
inline ThreadConfig ConfigureThreadConfig(int nthread) {
  if (nthread <= 0) {
    nthread = MaxNumThread();
    TREELITE_CHECK_GE(nthread, 1) << "Invalid number of threads configured in OpenMP";
  } else {
    TREELITE_CHECK_LE(nthread, MaxNumThread())
      << "nthread cannot exceed " << MaxNumThread() << " (configured by OpenMP).";
  }
  return ThreadConfig{static_cast<std::uint32_t>(nthread)};
}

/*!
 * \brief OpenMP scheduling directives
 */
struct ParallelSchedule {
  enum {
    kAuto,
    kDynamic,
    kStatic,
    kGuided,
  } sched;
  std::size_t chunk{0};

  ParallelSchedule static Auto() { return ParallelSchedule{kAuto}; }
  ParallelSchedule static Dynamic(std::size_t n = 0) { return ParallelSchedule{kDynamic, n}; }
  ParallelSchedule static Static(std::size_t n = 0) { return ParallelSchedule{kStatic, n}; }
  ParallelSchedule static Guided() { return ParallelSchedule{kGuided}; }
};

#if defined(_MSC_VER)
// msvc doesn't support unsigned integer as openmp index.
template <typename IndexType>
using OmpInd = std::conditional_t<std::is_signed<IndexType>::value, IndexType, std::int64_t>;
#else
template <typename IndexType>
using OmpInd = IndexType;
#endif

/*!
 * \brief In parallel, execute a function over a 1D range of elements.
 *
 * Schedule threads to call func(i, thread_id) in parallel, where:
 * - i is drawn from the range [begin, end)
 * - thread_id is the thread ID.
 * @param begin The beginning of the 1D range
 * @param end The end of the 1D range
 * @param thread_config Thread configuration
 * @param sched Scheduling directive to use
 * @param func Function to execute in parallel
 */
template <typename IndexType, typename FuncType>
inline void ParallelFor(IndexType begin, IndexType end, const ThreadConfig& thread_config,
                        ParallelSchedule sched, FuncType func) {
  if (begin == end) {
    return;
  }
  std::uint32_t nthread = thread_config.nthread;

  OMPException exc;
  switch (sched.sched) {
  case ParallelSchedule::kAuto: {
#pragma omp parallel for num_threads(nthread)
    for (OmpInd<IndexType> i = begin; i < end; ++i) {
      exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
    }
    break;
  }
  case ParallelSchedule::kDynamic: {
    if (sched.chunk == 0) {
#pragma omp parallel for num_threads(nthread) schedule(dynamic)
      for (OmpInd<IndexType> i = begin; i < end; ++i) {
        exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
      }
    } else {
#pragma omp parallel for num_threads(nthread) schedule(dynamic, sched.chunk)
      for (OmpInd<IndexType> i = begin; i < end; ++i) {
        exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
      }
    }
    break;
  }
  case ParallelSchedule::kStatic: {
    if (sched.chunk == 0) {
#pragma omp parallel for num_threads(nthread) schedule(static)
      for (OmpInd<IndexType> i = begin; i < end; ++i) {
        exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
      }
    } else {
#pragma omp parallel for num_threads(nthread) schedule(static, sched.chunk)
      for (OmpInd<IndexType> i = begin; i < end; ++i) {
        exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
      }
    }
    break;
  }
  case ParallelSchedule::kGuided: {
#pragma omp parallel for num_threads(nthread) schedule(guided)
    for (OmpInd<IndexType> i = begin; i < end; ++i) {
      exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
    }
    break;
  }
  }
  exc.Rethrow();
}

/*!
 * \brief In parallel, execute a function over a 2D range of elements.
 *
 * Schedule threads to call func(i, j, thread_id) in parallel, where:
 * - i is drawn from the range [dim1_begin, dim1_end)
 * - j is drawn from the range [dim2_begin, dim2_end)
 * - thread_id is the thread ID.
 * @param dim1_begin The beginning of the first axis of the 2D range
 * @param dim1_end The beginning of the first axis of the 2D range
 * @param dim2_begin The beginning of the second axis of the 2D range
 * @param dim2_end The beginning of the second axis of the 2D range
 * @param thread_config Thread configuration
 * @param sched Scheduling directive to use
 * @param func Function to execute in parallel
 */
template <typename IndexType1, typename IndexType2, typename FuncType>
inline void ParallelFor2D(IndexType1 dim1_begin, IndexType1 dim1_end,
                          IndexType2 dim2_begin, IndexType2 dim2_end,
                          const ThreadConfig& thread_config, ParallelSchedule sched,
                          FuncType func) {
  if (dim1_begin >= dim1_end || dim2_begin >= dim2_end) {   // degenerate range; do nothing
    return;
  }
  std::uint32_t nthread = thread_config.nthread;
  using IndexType =
    std::conditional_t<sizeof(IndexType1) >= sizeof(IndexType2), IndexType1, IndexType2>;

  IndexType1 dim1_size = dim1_end - dim1_begin;
  IndexType2 dim2_size = dim2_end - dim2_begin;
  IndexType size = dim1_size * dim2_size;

  OMPException exc;
  switch (sched.sched) {
    case ParallelSchedule::kAuto: {
#pragma omp parallel for num_threads(nthread)
      for (OmpInd<IndexType> k = 0; k < size; ++k) {
        auto i = dim1_begin + k / dim2_size;
        auto j = dim2_begin + k % dim2_size;
        exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                omp_get_thread_num());
      }
      break;
    }
    case ParallelSchedule::kDynamic: {
      if (sched.chunk == 0) {
#pragma omp parallel for num_threads(nthread) schedule(dynamic)
        for (OmpInd<IndexType> k = 0; k < size; ++k) {
          auto i = dim1_begin + k / dim2_size;
          auto j = dim2_begin + k % dim2_size;
          exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                  omp_get_thread_num());
        }
      } else {
#pragma omp parallel for num_threads(nthread) schedule(dynamic, sched.chunk)
        for (OmpInd<IndexType> k = 0; k < size; ++k) {
          auto i = dim1_begin + k / dim2_size;
          auto j = dim2_begin + k % dim2_size;
          exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                  omp_get_thread_num());
        }
      }
      break;
    }
    case ParallelSchedule::kStatic: {
      if (sched.chunk == 0) {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (OmpInd<IndexType> k = 0; k < size; ++k) {
          auto i = dim1_begin + k / dim2_size;
          auto j = dim2_begin + k % dim2_size;
          exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                  omp_get_thread_num());
        }
      } else {
#pragma omp parallel for num_threads(nthread) schedule(static, sched.chunk)
        for (OmpInd<IndexType> k = 0; k < size; ++k) {
          auto i = dim1_begin + k / dim2_size;
          auto j = dim2_begin + k % dim2_size;
          exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                  omp_get_thread_num());
        }
      }
      break;
    }
    case ParallelSchedule::kGuided: {
#pragma omp parallel for num_threads(nthread) schedule(guided)
      for (OmpInd<IndexType> k = 0; k < size; ++k) {
        auto i = dim1_begin + k / dim2_size;
        auto j = dim2_begin + k % dim2_size;
        exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                omp_get_thread_num());
      }
      break;
    }
  }
  exc.Rethrow();
}

}  // namespace threading_utils
}  // namespace treelite

#endif  // TREELITE_THREADING_UTILS_PARALLEL_FOR_H_
