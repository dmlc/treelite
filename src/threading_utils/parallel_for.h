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

inline int MaxNumThread() {
  return omp_get_max_threads();
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
inline void ParallelFor(IndexType begin, IndexType end, int nthread, ParallelSchedule sched,
                        FuncType func) {
  TREELITE_CHECK_GT(nthread, 0) << "nthread must be positive";
  TREELITE_CHECK_LE(nthread, MaxNumThread()) << "nthread cannot exceed " << MaxNumThread();
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
#pragma omp parallel for num_threads(nthread)
    for (OmpInd i = begin; i < end; ++i) {
      exc.Run(func, i, omp_get_thread_num());
    }
    break;
  }
  case ParallelSchedule::kDynamic: {
    if (sched.chunk == 0) {
#pragma omp parallel for num_threads(nthread) schedule(dynamic)
      for (OmpInd i = begin; i < end; ++i) {
        exc.Run(func, i, omp_get_thread_num());
      }
    } else {
#pragma omp parallel for num_threads(nthread) schedule(dynamic, sched.chunk)
      for (OmpInd i = begin; i < end; ++i) {
        exc.Run(func, i, omp_get_thread_num());
      }
    }
    break;
  }
  case ParallelSchedule::kStatic: {
    if (sched.chunk == 0) {
#pragma omp parallel for num_threads(nthread) schedule(static)
      for (OmpInd i = begin; i < end; ++i) {
        exc.Run(func, i, omp_get_thread_num());
      }
    } else {
#pragma omp parallel for num_threads(nthread) schedule(static, sched.chunk)
      for (OmpInd i = begin; i < end; ++i) {
        exc.Run(func, i, omp_get_thread_num());
      }
    }
    break;
  }
  case ParallelSchedule::kGuided: {
#pragma omp parallel for num_threads(nthread) schedule(guided)
    for (OmpInd i = begin; i < end; ++i) {
      exc.Run(func, i, omp_get_thread_num());
    }
    break;
  }
  }
  exc.Rethrow();
}

}  // namespace threading_utils
}  // namespace treelite

#endif  // TREELITE_THREADING_UTILS_PARALLEL_FOR_H_
