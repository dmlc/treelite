/*!
 * Copyright (c) 2021 by Contributors
 * \file thread_local.h
 * \brief Helper class for thread-local storage
 * \author Hyunsu Cho
 */
#ifndef TREELITE_THREAD_LOCAL_H_
#define TREELITE_THREAD_LOCAL_H_

namespace treelite {

/*!
 * \brief A thread-local storage
 * \tparam T the type we like to store
 */
template <typename T>
class ThreadLocalStore {
 public:
  /*! \return get a thread local singleton */
  static T* Get() {
    static thread_local T inst;
    return &inst;
  }
};

}  // namespace treelite

#endif  // TREELITE_THREAD_LOCAL_H_
