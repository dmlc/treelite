/*!
 * Copyright (c) 2023 by Contributors
 * \file pybuffer_frame.h
 * \brief Data structure to enable zero-copy exchange in Python
 * \author Hyunsu Cho
 */

#ifndef TREELITE_PYBUFFER_FRAME_H_
#define TREELITE_PYBUFFER_FRAME_H_

#include <cstddef>
#include <type_traits>

namespace treelite {

// Represent a frame in the Python buffer protocol (PEP 3118). We use a simplified representation
// to hold only 1-D arrays with stride 1.
struct PyBufferFrame {
  void* buf;
  char* format;
  std::size_t itemsize;
  std::size_t nitem;
};

static_assert(std::is_pod<PyBufferFrame>::value, "PyBufferFrame must be a POD type");

}  // namespace treelite

#endif  // TREELITE_PYBUFFER_FRAME_H_
