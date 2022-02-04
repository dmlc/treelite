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
  int nthread;
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
  return ThreadConfig{nthread};
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
inline void ParallelFor(IndexType begin, IndexType end, ThreadConfig thread_config,
                        ParallelSchedule sched, FuncType func) {
  if (begin == end) {
    return;
  }
  int nthread = thread_config.nthread;

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
                          ThreadConfig thread_config, ParallelSchedule sched, FuncType func) {
  if (dim1_begin >= dim1_end || dim2_begin >= dim2_end) {   // degenerate range; do nothing
    return;
  }
  int nthread = thread_config.nthread;

  OMPException exc;
  switch (sched.sched) {
    case ParallelSchedule::kAuto: {
#pragma omp parallel for num_threads(nthread) collapse(2) schedule(auto)
      for (OmpInd<IndexType1> i = dim1_begin; i < dim1_end; ++i) {
        for (OmpInd<IndexType2> j = dim2_begin; j < dim2_end; ++j) {
          exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                  omp_get_thread_num());
        }
      }
      break;
    }
    case ParallelSchedule::kDynamic: {
      if (sched.chunk == 0) {
#pragma omp parallel for num_threads(nthread) collapse(2) schedule(dynamic)
        for (OmpInd<IndexType1> i = dim1_begin; i < dim1_end; ++i) {
          for (OmpInd<IndexType2> j = dim2_begin; j < dim2_end; ++j) {
            exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                    omp_get_thread_num());
          }
        }
      } else {
#pragma omp parallel for num_threads(nthread) collapse(2) schedule(dynamic, sched.chunk)
        for (OmpInd<IndexType1> i = dim1_begin; i < dim1_end; ++i) {
          for (OmpInd<IndexType2> j = dim2_begin; j < dim2_end; ++j) {
            exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                    omp_get_thread_num());
          }
        }
      }
      break;
    }
    case ParallelSchedule::kStatic: {
      if (sched.chunk == 0) {
#pragma omp parallel for num_threads(nthread) collapse(2) schedule(static)
        for (OmpInd<IndexType1> i = dim1_begin; i < dim1_end; ++i) {
          for (OmpInd<IndexType2> j = dim2_begin; j < dim2_end; ++j) {
            exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                    omp_get_thread_num());
          }
        }
      } else {
#pragma omp parallel for num_threads(nthread) collapse(2) schedule(static, sched.chunk)
        for (OmpInd<IndexType1> i = dim1_begin; i < dim1_end; ++i) {
          for (OmpInd<IndexType2> j = dim2_begin; j < dim2_end; ++j) {
            exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                    omp_get_thread_num());
          }
        }
      }
      break;
    }
    case ParallelSchedule::kGuided: {
#pragma omp parallel for num_threads(nthread) collapse(2) schedule(guided)
      for (OmpInd<IndexType1> i = dim1_begin; i < dim1_end; ++i) {
        for (OmpInd<IndexType2> j = dim2_begin; j < dim2_end; ++j) {
          exc.Run(func, static_cast<IndexType1>(i), static_cast<IndexType2>(j),
                  omp_get_thread_num());
        }
      }
      break;
    }
  }
  exc.Rethrow();
}

}  // namespace threading_utils
}  // namespace treelite

#endif  // TREELITE_THREADING_UTILS_PARALLEL_FOR_H_
