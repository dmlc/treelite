/*!
 * Copyright (c) 2022 by Contributors
 * \file error.h
 * \brief Exception class used throughout the Treelite codebase
 * \author Hyunsu Cho
 */
#ifndef TREELITE_ERROR_H_
#define TREELITE_ERROR_H_

#include <stdexcept>
#include <string>

namespace treelite {

/*!
 * \brief Exception class that will be thrown by Treelite
 */
struct Error : public std::runtime_error {
  explicit Error(std::string const& s) : std::runtime_error(s) {}
};

}  // namespace treelite

#endif  // TREELITE_ERROR_H_
