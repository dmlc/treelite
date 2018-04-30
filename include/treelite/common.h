/*!
 * Copyright by 2017 Contributors
 * \file common.h
 * \brief Some useful utilities
 * \author Philip Cho
 */
#ifndef TREELITE_COMMON_H_
#define TREELITE_COMMON_H_

#include <treelite/base.h>
#include <dmlc/logging.h>
#include <dmlc/json.h>
#include <dmlc/data.h>
#include <algorithm>
#include <vector>
#include <limits>
#include <string>
#include <memory>
#include <string>
#include <sstream>
#include <iterator>
#include <functional>
#include <limits>
#include <iomanip>
#include <cerrno>
#include <climits>

namespace treelite {
namespace common {

/*! \brief abstract interface for classes that can be cloned */
class Cloneable {
 public:
  virtual ~Cloneable() = default;
  virtual Cloneable* clone() const = 0;  // for copy operation
  virtual Cloneable* move_clone() = 0;   // for move operation
};

/*! \brief macro to define boilerplate code for Cloneable classes */
#define CLONEABLE_BOILERPLATE(className) \
  explicit className(const className& other) = default;  \
  explicit className(className&& other) = default;  \
  Cloneable* clone() const override {  \
    return new className(*this);  \
  }  \
  Cloneable* move_clone() override {  \
    return new className(std::move(*this));  \
  }

/*!
 * \brief a wrapper around std::unique_ptr that supports deep copying and
 *        moving.
 */
template <typename T>
class DeepCopyUniquePtr {
 public:
  static_assert(std::is_base_of<Cloneable, T>::value,
                "DeepCopyUniquePtr requires a Cloneable type");
  ~DeepCopyUniquePtr() {}

  explicit DeepCopyUniquePtr(const T& other)
    : ptr(dynamic_cast<T*>(other.clone())) {}
    // downcasting is okay here because the other object is certainly of type T
  explicit DeepCopyUniquePtr(T&& other)
    : ptr(dynamic_cast<T*>(other.move_clone())) {}
  explicit DeepCopyUniquePtr(const DeepCopyUniquePtr<T>& other)
    : ptr(dynamic_cast<T*>(other.ptr->clone())) {}
  explicit DeepCopyUniquePtr(DeepCopyUniquePtr<T>&& other)
    : ptr(std::move(other.ptr)) {}

  inline T& operator*() {
    return *ptr;
  }
  const inline T& operator*() const {
    return *ptr;
  }
  T* operator->() {
    return ptr.operator->();
  }
  const T* operator->() const {
    return ptr.operator->();
  }

 private:
  std::unique_ptr<T> ptr;
};

/*!
 * \brief construct a new object of type T and wraps it with a std::unique_ptr.
 *        This is support legacy compiles (e.g. g++ 4.8.x) that do not yet
 *        support std::make_unique<T>.
 * \param args list of arguments with which an instance of T will be constructed
 * \return unique_ptr wrapping the newly created object
 * \tparam T type of object to be constructed
 * \tparam Args variadic template for forwarded arguments
 */
template<typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&& ...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/*!
 * \brief utility function to move an object pointed by a std::unique_ptr.
 *        After the move operation, the unique_ptr will hold an invalid object.
 * Usage example:
 * \code
 *   // Suppose class Foo has a move constructor with signature Foo(Bar&& bar)
 *   std::unique_ptr<Bar> bar;
 *   Foo foo(MoveUniquePtr(bar));
 * \endcode
 * \param ptr unique_ptr
 * \return rvalue reference to the object being moved
 * \tparam type of object being moved
 */
template <typename T>
inline T&& MoveUniquePtr(const std::unique_ptr<T>& ptr) {
  return std::move(*ptr.get());
}

/*! \brief format array as text, wrapped to a given maximum text width. Uses
 *         high precision to render floating-point values. */
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
      default_precision_(oss_.precision()), line_length_(indent) {
    oss_ << std::string(indent, ' ');
  }

  /*!
   * \brief add an entry (will use high precision for floating-point values)
   * \param e entry to be added
   */
  template <typename T>
  inline ArrayFormatter& operator<<(const T& e) {
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

/*!
 * \brief apply indentation to a multi-line string by inserting spaces at
 *        the beginning of each line
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
 * \brief perform binary search on the range [begin, end).
 * \param begin beginning of the search range
 * \param end end of the search range
 * \param val value being searched
 * \return iterator pointing to the value if found; end if value not found.
 * \tparam Iter type of iterator
 * \tparam T type of elements
 */
template<class Iter, class T>
Iter binary_search(Iter begin, Iter end, const T& val) {
  Iter i = std::lower_bound(begin, end, val);
  if (i != end && !(val < *i)) {
    return i;  // found
  } else {
    return end;  // not found
  }
}

/*!
 * \brief obtain a string representation of floating-point value, expressed
 * in high precision
 * \param value a value of primitive type
 * \return string representation
 */
template <typename T>
inline std::string ToStringHighPrecision(T value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<T>::digits10 + 2) << value;
  return oss.str();
}

/*!
 * \brief write a sequence of strings to a text file, with newline character
 *        (\n) inserted between strings. This function is suitable for creating
 *        multi-line text files.
 * \param filename name of text file
 * \param lines a sequence of strings to be written.
 */
inline void WriteToFile(const std::string& filename,
                        const std::string& content) {
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(filename.c_str(), "w"));
  dmlc::ostream os(fo.get());
  os << content;
  // force flush before fo destruct.
  os.set_stream(nullptr);
}

/*!
 * \brief apply a given transformation to a sequence of strings and append them
 *        to another sequence.
 * \param p_dest pointer to the destination sequence
 * \param lines a list of string lines to be transformed and then appended.
 * \param func transformation function.
 */
inline void TransformPushBack(std::vector<std::string>* p_dest,
                              const std::vector<std::string>& lines,
                              std::function<std::string(std::string)> func) {
  auto& dest = *p_dest;
  std::transform(lines.begin(), lines.end(), std::back_inserter(dest), func);
}

/*!
 * \brief convert text to number
 * \param str string containing number
 * \return number converted to type T
 * \tparam T type of value (should be a floating-point or integer type)
 */
template <typename T>
inline T TextToNumber(const std::string& str) {
  static_assert(std::is_same<T, float>::value
                || std::is_same<T, double>::value
                || std::is_same<T, int>::value
                || std::is_same<T, int8_t>::value
                || std::is_same<T, uint32_t>::value,
                "unsupported data type for TextToNumber; use float, double, "
                "int, int8_t, or uint32_t.");
}

template <>
inline float TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  float val = std::strtof(str.c_str(), &endptr);
  if (errno == ERANGE) {
    LOG(FATAL) << "Range error while converting string to double";
  } else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid floating-point number";
  }
  return val;
}

template <>
inline double TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  double val = std::strtod(str.c_str(), &endptr);
  if (errno == ERANGE) {
    LOG(FATAL) << "Range error while converting string to double";
  } else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid floating-point number";
  }
  return val;
}

template <>
inline int TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  auto val = std::strtol(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val < INT_MIN || val > INT_MAX) {
    LOG(FATAL) << "Range error while converting string to int";
  } else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<int>(val);
}

template <>
inline int8_t TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  auto val = std::strtol(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val < INT8_MIN || val > INT8_MAX) {
    LOG(FATAL) << "Range error while converting string to int8_t";
  } else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<int8_t>(val);
}

template <>
inline uint32_t TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  auto val = std::strtoul(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val > UINT32_MAX) {
    LOG(FATAL) << "Range error while converting string to uint32_t";
  } else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<uint32_t>(val);
}

/*!
 * \brief convert text to number array
 * \param str string containing numbers, separated by spaces in between
 * \return std::vector of numbers, converted to type T
 * \tparam T type of value (should be a floating-point or integer type)
 */
template <typename T>
inline std::vector<T> TextToArray(const std::string& text, int num_entry) {
  std::vector<T> array;
  std::istringstream ss(text);
  std::string token;
  for (int i = 0; i < num_entry; ++i) {
    std::getline(ss, token, ' ');
    array.push_back(TextToNumber<T>(token));
  }
  return array;
}

/*!
 * \brief split text using a delimiter
 * \param str text
 * \param delim delimiter
 * \return std::vector of strings, split by a delimiter
 */
inline std::vector<std::string> Split(const std::string& text, char delim) {
  std::vector<std::string> array;
  std::istringstream ss(text);
  std::string token;
  while (std::getline(ss, token, delim)) {
    array.push_back(token);
  }
  return array;
}

/*!
 * \brief perform comparison between two float's using a comparsion operator
 * The comparison will be in the form [lhs] [op] [rhs].
 * \param lhs float on the left hand side
 * \param op comparison operator
 * \param rhs float on the right hand side
 * \return whether [lhs] [op] [rhs] is true or not
 */
inline bool CompareWithOp(treelite::tl_float lhs, treelite::Operator op,
                          treelite::tl_float rhs) {
  switch (op) {
    case treelite::Operator::kEQ: return lhs == rhs;
    case treelite::Operator::kLT: return lhs <  rhs;
    case treelite::Operator::kLE: return lhs <= rhs;
    case treelite::Operator::kGT: return lhs >  rhs;
    case treelite::Operator::kGE: return lhs >= rhs;
    default:            LOG(FATAL) << "operator undefined";
  }
}

}  // namespace common
}  // namespace treelite
#endif  // TREELITE_COMMON_H_
