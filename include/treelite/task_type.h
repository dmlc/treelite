/*!
 * Copyright (c) 2023 by Contributors
 * \file task_type.h
 * \brief An enum type to indicate the learning task.
 * \author Hyunsu Cho
 */

#ifndef TREELITE_TASK_TYPE_H_
#define TREELITE_TASK_TYPE_H_

#include <treelite/logging.h>

#include <cstdint>
#include <string>

namespace treelite {

/*!
 * \brief Enum type representing the task type.
 *
 * The task type places constraints on the parameters of TaskParam. See the docstring for each
 * enum constants for more details.
 */
enum class TaskType : std::uint8_t {
  /*!
   * \brief Catch-all task type encoding all tasks that are not multi-class classification, such as
   *        binary classification, regression, and learning-to-rank.
   *
   * The kBinaryClfRegr task type implies the following constraints on the task parameters:
   * output_type=float, grove_per_class=false, num_class=1, leaf_vector_size=1.
   */
  kBinaryClfRegr = 0,
  /*!
   * \brief The multi-class classification task, in which the prediction for each class is given
   *        by the sum of outputs from a subset of the trees. We refer to this method as
   *        "grove-per-class".
   *
   * In this setting, each leaf node in a tree produces a single scalar output. To obtain
   * predictions for each class, we divide the trees into multiple groups ("groves") and then
   * compute the sum of outputs of the trees in each group. The prediction for the i-th class is
   * given by the sum of the outputs of the trees whose index is congruent to [i] modulo
   * [num_class].
   *
   * Examples of "grove-per-class" classifier are found in XGBoost, LightGBM, and
   * GradientBoostingClassifier of scikit-learn.
   *
   * The kMultiClfGrovePerClass task type implies the following constraints on the task parameters:
   * output_type=float, grove_per_class=true, num_class>1, leaf_vector_size=1. In addition, we
   * require that the number of trees is evenly divisible by [num_class].
   */
  kMultiClfGrovePerClass = 1,
  /*!
   * \brief The multi-class classification task, in which each tree produces a vector of
   *        probability predictions for all the classes.
   *
   * In this setting, each leaf node in a tree produces a vector output whose length is [num_class].
   * The vector represents probability predictions for all the classes. The outputs of the trees
   * are combined via summing or averaging, depending on the value of the [average_tree_output]
   * field. In effect, each tree is casting a set of weighted (fractional) votes for the classes.
   *
   * Examples of kMultiClfProbDistLeaf task type are found in RandomForestClassifier of
   * scikit-learn and RandomForestClassifier of cuML.
   *
   * The kMultiClfProbDistLeaf task type implies the following constraints on the task parameters:
   * output_type=float, grove_per_class=false, num_class>1, leaf_vector_size=num_class.
   */
  kMultiClfProbDistLeaf = 2,
  /*!
   * \brief The multi-class classification task, in which each tree produces a single integer output
   *        representing an unweighted vote for a particular class.
   *
   * In this setting, each leaf node in a tree produces a single integer output between 0 and
   * [num_class-1] that indicates a vote for a particular class. The outputs of the trees are
   * combined by summing one_hot(tree(i)), where one_hot(x) represents the one-hot-encoded vector
   * with 1 in index [x] and 0 everywhere else, and tree(i) is the output from the i-th tree.
   * Models of type kMultiClfCategLeaf can be converted into the kMultiClfProbDistLeaf type, by
   * converting the output of every leaf node into the equivalent one-hot-encoded vector.
   *
   * The kMultiClfCategLeaf task type implies the following constraints on the task parameters:
   * output_type=int, grove_per_class=false, num_class>1, leaf_vector_size=1.
   */
  kMultiClfCategLeaf = 3
};

inline std::string TaskTypeToString(TaskType type) {
  switch (type) {
  case TaskType::kBinaryClfRegr:
    return "kBinaryClfRegr";
  case TaskType::kMultiClfGrovePerClass:
    return "kMultiClfGrovePerClass";
  case TaskType::kMultiClfProbDistLeaf:
    return "kMultiClfProbDistLeaf";
  case TaskType::kMultiClfCategLeaf:
    return "kMultiClfCategLeaf";
  default:
    return "";
  }
}

inline TaskType StringToTaskType(std::string const& str) {
  if (str == "kBinaryClfRegr") {
    return TaskType::kBinaryClfRegr;
  } else if (str == "kMultiClfGrovePerClass") {
    return TaskType::kMultiClfGrovePerClass;
  } else if (str == "kMultiClfProbDistLeaf") {
    return TaskType::kMultiClfProbDistLeaf;
  } else if (str == "kMultiClfCategLeaf") {
    return TaskType::kMultiClfCategLeaf;
  } else {
    TREELITE_LOG(FATAL) << "Unknown task type: " << str;
    return TaskType::kBinaryClfRegr;  // to avoid compiler warning
  }
}

}  // namespace treelite

#endif  // TREELITE_TASK_TYPE_H_
