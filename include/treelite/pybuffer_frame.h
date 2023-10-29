/*!
 * Copyright (c) 2023 by Contributors
 * \file pybuffer_frame.h
 * \brief Data structure to enable zero-copy exchange in Python
 * \author Hyunsu Cho
 */

#ifndef TREELITE_PYBUFFER_FRAME_H_
#define TREELITE_PYBUFFER_FRAME_H_

#include <treelite/c_api.h>

#include <cstddef>
#include <type_traits>

namespace treelite {

using PyBufferFrame = TreelitePyBufferFrame;

static_assert(std::is_pod<PyBufferFrame>::value, "PyBufferFrame must be a POD type");

}  // namespace treelite

#endif  // TREELITE_PYBUFFER_FRAME_H_
