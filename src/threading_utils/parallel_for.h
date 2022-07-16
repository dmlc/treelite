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
#include <treelite/omp_exception.h>
#include <type_traits>
#include <algorithm>
#include <exception>
#include <mutex>
#include <cstddef>
#include <cstdint>

namespace treelite {
namespace threading_utils {

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

// OpenMP schedule
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

template <typename IndexType, typename FuncType>
inline void ParallelFor(IndexType begin, IndexType end, const ThreadConfig& thread_config,
                        ParallelSchedule sched, FuncType func) {
  if (begin == end) {
    return;
  }

#if defined(_MSC_VER)
  // msvc doesn't support unsigned integer as openmp index.
  using OmpInd = std::conditional_t<std::is_signed<IndexType>::value, IndexType, std::int64_t>;
#else
  using OmpInd = IndexType;
#endif

  OMPException exc;
  switch (sched.sched) {
  case ParallelSchedule::kAuto: {
#pragma omp parallel for num_threads(thread_config.nthread)
    for (OmpInd i = begin; i < end; ++i) {
      exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
    }
    break;
  }
  case ParallelSchedule::kDynamic: {
    if (sched.chunk == 0) {
#pragma omp parallel for num_threads(thread_config.nthread) schedule(dynamic)
      for (OmpInd i = begin; i < end; ++i) {
        exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
      }
    } else {
#pragma omp parallel for num_threads(thread_config.nthread) schedule(dynamic, sched.chunk)
      for (OmpInd i = begin; i < end; ++i) {
        exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
      }
    }
    break;
  }
  case ParallelSchedule::kStatic: {
    if (sched.chunk == 0) {
#pragma omp parallel for num_threads(thread_config.nthread) schedule(static)
      for (OmpInd i = begin; i < end; ++i) {
        exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
      }
    } else {
#pragma omp parallel for num_threads(thread_config.nthread) schedule(static, sched.chunk)
      for (OmpInd i = begin; i < end; ++i) {
        exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
      }
    }
    break;
  }
  case ParallelSchedule::kGuided: {
#pragma omp parallel for num_threads(thread_config.nthread) schedule(guided)
    for (OmpInd i = begin; i < end; ++i) {
      exc.Run(func, static_cast<IndexType>(i), omp_get_thread_num());
    }
    break;
  }
  }
  exc.Rethrow();
}

}  // namespace threading_utils
}  // namespace treelite

#endif  // TREELITE_THREADING_UTILS_PARALLEL_FOR_H_
