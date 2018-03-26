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
#include <memory>
#include <iterator>
#include <functional>
#include <iomanip>
#include <cerrno>
#include <climits>

namespace treelite {
namespace common {

/*! \brief abstract interface for classes that can be cloned */
class Cloneable {
 public:
  virtual ~Cloneable() = default;
  virtual Cloneable* clone() const = 0; // for copy operation
  virtual Cloneable* move_clone() = 0;  // for move operation
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

/*!
 * \brief insert a string into a string stream, adding a newline character
 *        (\n) so that the string stream has no line longer than the given
 *        width.
 * \param p_stm pointer to string stream
 * \param p_length used to keep track of the lenght of current line. It will
 *                 always be the case that [length] <= [textwidth]
 * \param indent indent level inside array
 * \param str string to be inserted
 * \param textwidth maximum width of each line
 */
inline void WrapText(std::ostringstream* p_stm, size_t* p_length,
                     const std::string& str, size_t indent,
                     size_t textwidth) {
  std::ostringstream& stm = *p_stm;
  size_t& length = *p_length;
  if (length + str.length() <= textwidth) {
    stm << str;
    length += str.length();
  } else {
    stm << "\n" << std::string(indent, ' ') << str;
    length = str.length() + indent;
  }
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
Iter binary_search(Iter begin, Iter end, const T& val)
{
  Iter i = std::lower_bound(begin, end, val);
  if (i != end && !(val < *i)) {
    return i;  // found
  } else {
    return end;  // not found
  }
}

/*!
 * \brief obtain a string representation of primitive type using ostringstream
 * \param value a value of primitive type
 * \return string representation
 */
template <typename T>
inline std::string ToString(T value) {
  std::ostringstream oss;
  // to restore default precision
  const std::streamsize ss = std::cout.precision();
  oss << std::setprecision(std::numeric_limits<T>::digits10 + 2) << value
      << std::setprecision(ss);
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
  }
  else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  }
  else if (*endptr != '\0') {
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
  }
  else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  }
  else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid floating-point number";
  }
  return val;
}

template <>
inline int TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  long int val = std::strtol(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val < INT_MIN || val > INT_MAX) {
    LOG(FATAL) << "Range error while converting string to int";
  }
  else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  }
  else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<int>(val);
}

template <>
inline int8_t TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  long int val = std::strtol(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val < INT8_MIN || val > INT8_MAX) {
    LOG(FATAL) << "Range error while converting string to int8_t";
  }
  else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  }
  else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<int8_t>(val);
}

template <>
inline uint32_t TextToNumber(const std::string& str) {
  static_assert(sizeof(uint32_t) <= sizeof(unsigned long int),
    "unsigned long int too small to hold uint32_t");
  errno = 0;
  char *endptr;
  unsigned long int val = std::strtoul(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val > UINT32_MAX) {
    LOG(FATAL) << "Range error while converting string to uint32_t";
  }
  else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  }
  else if (*endptr != '\0') {
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
  switch(op) {
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
