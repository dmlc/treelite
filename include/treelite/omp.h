/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file omp.h
 * \brief compatiblity wrapper for systems that don't support OpenMP
 * \author Hyunsu Cho
 */
#ifndef TREELITE_OMP_H_
#define TREELITE_OMP_H_

#ifdef TREELITE_OPENMP_SUPPORT
#include <omp.h>

#include <limits>

// MSVC doesn't implement the thread limit.
#if defined(_MSC_VER)
inline int omp_get_thread_limit() {
  return std::numeric_limits<int>::max();
}
#endif  // defined(_MSC_VER)

#else  // TREELITE_OPENMP_SUPPORT

// Stubs for OpenMP functions, to be used when OpenMP is not available.

inline int omp_get_thread_limit() {
  return 1;
}

inline int omp_get_thread_num() {
  return 0;
}

inline int omp_get_max_threads() {
  return 1;
}

inline int omp_get_num_procs() {
  return 1;
}

#endif  // TREELITE_OPENMP_SUPPORT

#endif  // TREELITE_OMP_H_
