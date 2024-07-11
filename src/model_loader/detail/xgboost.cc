/*!
 * Copyright (c) 2023 by Contributors
 * \file xgboost.cc
 * \brief Utility functions for XGBoost frontend
 * \author Hyunsu Cho
 */
#include "./xgboost.h"

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include <treelite/detail/file_utils.h>
#include <treelite/logging.h>
#include <treelite/model_loader.h>

namespace treelite::model_loader {

namespace detail::xgboost {

// Get correct postprocessor for prediction, depending on objective function
std::string GetPostProcessor(std::string const& objective_name) {
  std::vector<std::string> const exponential_objectives{
      "count:poisson", "reg:gamma", "reg:tweedie", "survival:cox", "survival:aft"};
  if (objective_name == "multi:softmax" || objective_name == "multi:softprob") {
    return "softmax";
  } else if (objective_name == "reg:logistic" || objective_name == "binary:logistic") {
    return "sigmoid";
  } else if (std::find(
                 exponential_objectives.cbegin(), exponential_objectives.cend(), objective_name)
             != exponential_objectives.cend()) {
    return "exponential";
  } else if (objective_name == "binary:hinge") {
    return "hinge";
  } else if (objective_name == "reg:squarederror" || objective_name == "reg:linear"
             || objective_name == "reg:squaredlogerror" || objective_name == "reg:pseudohubererror"
             || objective_name == "binary:logitraw" || objective_name == "rank:pairwise"
             || objective_name == "rank:ndcg" || objective_name == "rank:map") {
    return "identity";
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized XGBoost objective: " << objective_name;
    return "";
  }
}

double TransformBaseScoreToMargin(std::string const& postprocessor, double base_score) {
  if (postprocessor == "sigmoid") {
    return ProbToMargin::Sigmoid(base_score);
  } else if (postprocessor == "exponential") {
    return ProbToMargin::Exponential(base_score);
  } else {
    return base_score;
  }
}

}  // namespace detail::xgboost

std::string DetectXGBoostFormat(std::string const& filename) {
  constexpr std::size_t nbytes = 2;
  char buf[nbytes] = {0};

  std::ifstream ifs = treelite::detail::OpenFileForReadAsStream(filename);
  ifs.read(buf, nbytes);

  auto is_space = [](char c) -> bool { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; };

  // First look at the first character
  if (buf[0] == 'N') {
    // The no-op code is only used in UBJSON
    return "ubjson";
  } else if (is_space(buf[0])) {
    // White-spaces are only present in JSON
    return "json";
  } else if (buf[0] != '{') {
    // Otherwise, should have '{' if the file is JSON or UBJSON.
    return "unknown";
  }

  // First character is '{'. Now look at the second character.
  if (is_space(buf[1]) || buf[1] == '"') {
    // White-spaces and double quotation marks are only present in JSON
    return "json";
  } else if (buf[1] == 'N' || buf[1] == '$' || buf[1] == '#' || buf[1] == 'i' || buf[1] == 'U'
             || buf[1] == 'I' || buf[1] == 'l' || buf[1] == 'L') {
    // The no-op code and type markers are only present in UBJSON
    return "ubjson";
  }

  return "unknown";
}

}  // namespace treelite::model_loader
