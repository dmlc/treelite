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
#include <cfloat>
#include <cmath>

#ifndef _WIN32
#include <libgen.h>
#include <cstring>
#else
#include <cstdlib>
#endif

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
std::unique_ptr<T> make_unique(Args&& ...args)
{
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
 * \param str string to be inserted
 * \param textwidth maximum width of each line
 */
inline void WrapText(std::ostringstream* p_stm, size_t* p_length,
                     const std::string& str, size_t textwidth) {
  std::ostringstream& stm = *p_stm;
  size_t& length = *p_length;
  if (length + str.length() + 2 <= textwidth) {
    stm << str << ", ";
    length += str.length() + 2;
  } else {
    stm << "\n  " << str << ", ";
    length = str.length() + 4;
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
 * \brief obtain a string representation of a floating-point number
 * \param value floating-point number
 * \return string representation
 */
inline std::string FloatToString(tl_float value) {
  std::ostringstream oss;
  oss << value;
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
                        const std::vector<std::string>& lines) {
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(filename.c_str(), "w"));
  dmlc::ostream os(fo.get());
  std::copy(lines.begin(), lines.end(),
            std::ostream_iterator<std::string>(os, "\n"));
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
 * \brief check for NaN (Not a Number)
 * \param value value to check
 * \return whether the given value is NaN or not
 * \tparam type of value (should be a floating-point value)
 */
template <typename T>
inline bool CheckNAN(T value) {
#ifdef _MSC_VER
  return (_isnan(value) != 0);
#else
  return std::isnan(value);
#endif
}

/*!
 * \brief extract the base name from a full path. The base name is defined as
 *        the component that follows the last '/' in the full path.
 * \code
 *   GetBaseName("./food/bar.txt");  // returns bar.txt
 * \endcode
 * \param path full path
 */
#ifndef _WIN32
// basename for UNIX-like systems
inline std::string GetBasename(const std::string& path) {
  char* path_ = strdup(path.c_str());
  char* base = basename(path_);
  std::string ret(base);
  free(path_);
  return ret;
}
#else
// basename for Windows
inline std::string GetBasename(const std::string& path) {
  std::vector<char> fname(path.length() + 1);
  std::vector<char> ext(path.length() + 1);
  _splitpath_s(path.c_str(), NULL, 0, NULL, 0,
      &fname[0], path.length() + 1, &ext[0], path.length() + 1);
  return std::string(&fname[0]) + std::string(&ext[0]);
}
#endif

}  // namespace common
}  // namespace treelite
#endif  // TREELITE_COMMON_H_
