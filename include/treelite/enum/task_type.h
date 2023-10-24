/*!
 * Copyright (c) 2023 by Contributors
 * \file task_type.h
 * \brief Define enum type TaskType
 * \author Hyunsu Cho
 */

#ifndef TREELITE_ENUM_TASK_TYPE_H_
#define TREELITE_ENUM_TASK_TYPE_H_

#include <cstdint>
#include <string>

namespace treelite {

/*!
 * \brief Enum type representing the task type.
 */
enum class TaskType : std::uint8_t {
  kBinaryClf = 0,
  kRegressor = 1,
  kMultiClf = 2,
  kLearningToRank = 3,
  kIsolationForest = 4
};

/*! \brief Get string representation of TaskType */
std::string TaskTypeToString(TaskType type);

/*! \brief Get TaskType from string */
TaskType TaskTypeFromString(std::string const& str);

}  // namespace treelite

#endif  // TREELITE_ENUM_TASK_TYPE_H_
