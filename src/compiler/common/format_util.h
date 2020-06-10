/*!
 * Copyright (c) by 2020 Contributors
 * \file format_util.h
 * \brief Formatting utilities
 * \author Hyunsu Cho
 */
#ifndef TREELITE_COMPILER_COMMON_FORMAT_UTIL_H_
#define TREELITE_COMPILER_COMMON_FORMAT_UTIL_H_

#include <fmt/format.h>
#include <limits>
#include <string>
#include <sstream>
#include <iomanip>

namespace treelite {
namespace compiler {
namespace common_util {

/*!
 * \brief apply indentation to a multi-line string by inserting spaces at the beginning of each line
 * \param str multi-line string
 * \param indent indent level to be applied (in number of spaces)
 * \return indented string
 */
inline std::string IndentMultiLineString(const std::string& str,
                                         size_t indent = 2) {
  std::ostringstream oss;
  if (str[0] != '\n') {
    oss << std::string(indent, ' ');
  }
  bool need_indent = false;
    // one or more newlines will cause empty spaces to be inserted as indent
  for (char c : str) {  // assume UNIX-style line ending
    if (c == '\n') {
      need_indent = true;
    } else if (need_indent) {
      oss << std::string(indent, ' ');
      need_indent = false;
    }
    oss << c;
  }
  return oss.str();
}

/*!
 * \brief obtain a string representation of floating-point value, expressed
 * in high precision
 * \param value a value of primitive type
 * \return string representation
 */
template <typename T>
inline std::string ToStringHighPrecision(T value) {
  return fmt::format("{:.{}g}", value, std::numeric_limits<T>::digits10 + 2);
}

/*! \brief format array as text, wrapped to a given maximum text width. Uses high precision to
 *         render floating-point values. */
class ArrayFormatter {
 public:
  /*!
   * \brief constructor
   * \param text_width maximum text width
   * \param indent indentation level
   * \param delimiter delimiter between elements
   */
  ArrayFormatter(size_t text_width, size_t indent, char delimiter = ',')
    : oss_(), text_width_(text_width), indent_(indent), delimiter_(delimiter),
      default_precision_(oss_.precision()), line_length_(indent),
      is_empty_(true) {}

  /*!
   * \brief add an entry (will use high precision for floating-point values)
   * \param e entry to be added
   */
  template <typename T>
  inline ArrayFormatter& operator<<(const T& e) {
    if (is_empty_) {
      is_empty_ = false;
      oss_ << std::string(indent_, ' ');
    }
    std::ostringstream tmp;
    tmp << std::setprecision(GetPrecision<T>()) << e << delimiter_ << " ";
    const std::string token = tmp.str();  // token to be added to wrapped text
    if (line_length_ + token.length() <= text_width_) {
      oss_ << token;
      line_length_ += token.length();
    } else {
      oss_ << "\n" << std::string(indent_, ' ') << token;
      line_length_ = token.length() + indent_;
    }
    return *this;
  }

  /*!
   * \brief obtain formatted text containing the rendered array
   * \return string representing the rendered array
   */
  inline std::string str() {
    return oss_.str();
  }

 private:
  std::ostringstream oss_;  // string stream to store wrapped text
  const size_t indent_;  // indent level, to indent each line
  const size_t text_width_;  // maximum length of each line
  const char delimiter_;  // delimiter (defaults to comma)
  const int default_precision_;  // default precision used by string stream
  size_t line_length_;  // width of current line
  bool is_empty_;  // true if no entry has been added yet

  template <typename T>
  inline int GetPrecision() {
    return default_precision_;
  }
};

template <>
inline int ArrayFormatter::GetPrecision<float>() {
  return std::numeric_limits<float>::digits10 + 2;
}
template <>
inline int ArrayFormatter::GetPrecision<double>() {
  return std::numeric_limits<double>::digits10 + 2;
}

}  // namespace common_util
}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_COMMON_FORMAT_UTIL_H_
