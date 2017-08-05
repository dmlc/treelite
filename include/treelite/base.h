/*!
 * Copyright 2017 by Contributors
 * \file base.h
 * \brief defines configuration macros of treelite
 * \author Philip Cho
 */
#ifndef TREELITE_BASE_H_
#define TREELITE_BASE_H_

#include <cstdint>

namespace treelite {

typedef double tl_float;
/*! \brief comparison operators */
enum class Operator : int8_t {
  kEQ, kLT, kLE, kGT, kGE  // ==, <, <=, >, >=
};

}  // namespace treelite

#endif  // TREELITE_BASE_H_ 
