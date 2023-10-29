/*!
 * Copyright (c) 2023 by Contributors
 * \file c_api_utils.h
 * \author Hyunsu Cho
 * \brief C API of Treelite, used for interfacing with other languages
 */
#ifndef SRC_C_API_C_API_UTILS_H_
#define SRC_C_API_C_API_UTILS_H_

#include <treelite/pybuffer_frame.h>
#include <treelite/thread_local.h>

#include <cstdint>
#include <string>
#include <vector>

namespace treelite::c_api {

/*! \brief When returning a complex object from a C API function, we
 *         store the object here and then return a pointer. The
 *         storage is thread-local static storage. */
struct ReturnValueEntry {
  std::string ret_str;
  std::vector<std::uint64_t> ret_shape;
  std::vector<treelite::PyBufferFrame> ret_frames;
};
using ReturnValueStore = ThreadLocalStore<ReturnValueEntry>;

}  // namespace treelite::c_api

#endif  // SRC_C_API_C_API_UTILS_H_
