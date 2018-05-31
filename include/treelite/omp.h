/*!
 * Copyright 2017 by Contributors
 * \file omp.h
 * \brief compatiblity wrapper for systems that don't support OpenMP
 * \author Philip Cho
 */
#ifndef TREELITE_OMP_H_
#define TREELITE_OMP_H_

#ifdef TREELITE_OPENMP_SUPPORT
#include <omp.h>
#else
inline int omp_get_thread_num() {
  return 0;
}

inline int omp_get_max_threads() {
  return 1;
}
#endif  // TREELITE_OPENMP_SUPPORT

#endif  // TREELITE_OMP_H_
