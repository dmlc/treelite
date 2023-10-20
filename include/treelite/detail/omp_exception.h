/*!
 * Copyright (c) 2022-2023 by Contributors
 * \file omp_exception.h
 * \author Hyunsu Cho
 * \brief Utility to propagate exceptions throws inside an OpenMP block
 */
#ifndef TREELITE_DETAIL_OMP_EXCEPTION_H_
#define TREELITE_DETAIL_OMP_EXCEPTION_H_

#include <treelite/error.h>

#include <exception>
#include <mutex>

namespace treelite {

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
    } catch (treelite::Error& ex) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!omp_exception_) {
        omp_exception_ = std::current_exception();
      }
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

}  // namespace treelite

#endif  // TREELITE_DETAIL_OMP_EXCEPTION_H_
