/*!
 * Copyright (c) 2023 by Contributors
 * \file task_type.cc
 * \author Hyunsu Cho
 * \brief Utilities for TaskType enum
 */

#include <treelite/enum/task_type.h>
#include <treelite/logging.h>

#include <string>

namespace treelite {

std::string TaskTypeToString(TaskType type) {
  switch (type) {
  case TaskType::kBinaryClf:
    return "kBinaryClf";
  case TaskType::kRegressor:
    return "kRegressor";
  case TaskType::kMultiClf:
    return "kMultiClf";
  case TaskType::kLearningToRank:
    return "kLearningToRank";
  case TaskType::kIsolationForest:
    return "kIsolationForest";
  default:
    return "";
  }
}

TaskType TaskTypeFromString(std::string const& str) {
  if (str == "kBinaryClf") {
    return TaskType::kBinaryClf;
  } else if (str == "kRegressor") {
    return TaskType::kRegressor;
  } else if (str == "kMultiClf") {
    return TaskType::kMultiClf;
  } else if (str == "kLearningToRank") {
    return TaskType::kLearningToRank;
  } else if (str == "kIsolationForest") {
    return TaskType::kIsolationForest;
  } else {
    TREELITE_LOG(FATAL) << "Unknown task type: " << str;
    return TaskType::kBinaryClf;  // to avoid compiler warning
  }
}

}  // namespace treelite
