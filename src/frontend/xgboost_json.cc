/*!
 * Copyright (c) 2020-2021 by Contributors
 * \file xgboost_json.cc
 * \brief Frontend for xgboost model
 * \author Hyunsu Cho
 * \author William Hicks
 */

#include "xgboost/xgboost_json.h"

#include <fmt/format.h>
#include <rapidjson/error/en.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <treelite/math.h>
#include <treelite/logging.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>

#include "xgboost/xgboost.h"

namespace {

template <typename StreamType, typename ErrorHandlerFunc>
std::unique_ptr<treelite::Model> ParseStream(std::unique_ptr<StreamType> input_stream,
                                             ErrorHandlerFunc error_handler);

}  // anonymous namespace

namespace treelite {
namespace frontend {

std::unique_ptr<treelite::Model> LoadXGBoostJSONModel(const char* filename) {
  char read_buffer[65536];

#ifdef _WIN32
  FILE* fp = std::fopen(filename, "rb");
#else
  FILE* fp = std::fopen(filename, "r");
#endif
  if (!fp) {
    TREELITE_LOG(FATAL) << "Failed to open file '" << filename << "': " << std::strerror(errno);
  }

  auto input_stream = std::make_unique<rapidjson::FileReadStream>(
      fp, read_buffer, sizeof(read_buffer));
  auto error_handler = [fp](size_t offset) -> std::string {
    size_t cur = (offset >= 50 ? (offset - 50) : 0);
    std::fseek(fp, cur, SEEK_SET);
    int c;
    std::ostringstream oss, oss2;
    for (int i = 0; i < 100; ++i) {
      c = std::fgetc(fp);
      if (c == EOF) {
        break;
      }
      oss << static_cast<char>(c);
      if (cur == offset) {
        oss2 << "^";
      } else {
        oss2 << "~";
      }
      ++cur;
    }
    std::fclose(fp);
    return oss.str() + "\n" + oss2.str();
  };
  auto parsed_model = ParseStream(std::move(input_stream), error_handler);
  std::fclose(fp);
  return parsed_model;
}

std::unique_ptr<treelite::Model> LoadXGBoostJSONModelString(const char* json_str, size_t length) {
  auto input_stream = std::make_unique<rapidjson::MemoryStream>(json_str, length);
  auto error_handler = [json_str](size_t offset) -> std::string {
    size_t cur = (offset >= 50 ? (offset - 50) : 0);
    std::ostringstream oss, oss2;
    for (int i = 0; i < 100; ++i) {
      if (!json_str[cur]) {
        break;
      }
      oss << json_str[cur];
      if (cur == offset) {
        oss2 << "^";
      } else {
        oss2 << "~";
      }
      ++cur;
    }
    return oss.str() + "\n" + oss2.str();
  };
  return ParseStream(std::move(input_stream), error_handler);
}

}  // namespace frontend

namespace details {

/******************************************************************************
 * BaseHandler
 * ***************************************************************************/

bool BaseHandler::pop_handler() {
  if (auto parent = delegator.lock()) {
    parent->pop_delegate();
    return true;
  } else {
    return false;
  }
}

void BaseHandler::set_cur_key(const char *str, std::size_t length) {
  cur_key = std::string{str, length};
}

const std::string &BaseHandler::get_cur_key() { return cur_key; }

bool BaseHandler::check_cur_key(const std::string &query_key) {
  return cur_key == query_key;
}

template <typename ValueType>
bool BaseHandler::assign_value(const std::string &key,
                               ValueType &&value,
                               ValueType &output) {
  if (check_cur_key(key)) {
    output = value;
    return true;
  } else {
    return false;
  }
}

template <typename ValueType>
bool BaseHandler::assign_value(const std::string &key,
                  const ValueType &value,
                  ValueType &output) {
  if (check_cur_key(key)) {
    output = value;
    return true;
  } else {
    return false;
  }
}

/******************************************************************************
 * IgnoreHandler
 * ***************************************************************************/
bool IgnoreHandler::Null() { return true; }
bool IgnoreHandler::Bool(bool) { return true; }
bool IgnoreHandler::Int(int) { return true; }
bool IgnoreHandler::Uint(unsigned) { return true; }
bool IgnoreHandler::Int64(int64_t) { return true; }
bool IgnoreHandler::Uint64(uint64_t) { return true; }
bool IgnoreHandler::Double(double) { return true; }
bool IgnoreHandler::String(const char *, std::size_t, bool) {
  return true; }
bool IgnoreHandler::StartObject() { return push_handler<IgnoreHandler>(); }
bool IgnoreHandler::Key(const char *, std::size_t, bool) {
  return true; }
bool IgnoreHandler::StartArray() { return push_handler<IgnoreHandler>(); }

/******************************************************************************
 * TreeParamHandler
 * ***************************************************************************/
bool TreeParamHandler::String(const char *str, std::size_t, bool) {
  // Key "num_deleted" deprecated but still present in some xgboost output
  return (check_cur_key("num_feature") ||
          assign_value("num_nodes", std::atoi(str), output) ||
          check_cur_key("size_leaf_vector") || check_cur_key("num_deleted"));
}

/******************************************************************************
 * RegTreeHandler
 * ***************************************************************************/
bool RegTreeHandler::StartArray() {
  /* Keys "categories" and "split_type" not currently documented in schema but
   * will be used for upcoming categorical split feature */
  return (
      push_key_handler<ArrayHandler<double>>("loss_changes", loss_changes) ||
      push_key_handler<ArrayHandler<double>>("sum_hessian", sum_hessian) ||
      push_key_handler<ArrayHandler<double>>("base_weights", base_weights) ||
      push_key_handler<ArrayHandler<int>>("categories_segments", categories_segments) ||
      push_key_handler<ArrayHandler<int>>("categories_sizes", categories_sizes) ||
      push_key_handler<ArrayHandler<int>>("categories_nodes", categories_nodes) ||
      push_key_handler<ArrayHandler<int>>("categories", categories) ||
      push_key_handler<IgnoreHandler>("leaf_child_counts") ||
      push_key_handler<ArrayHandler<int>>("left_children", left_children) ||
      push_key_handler<ArrayHandler<int>>("right_children", right_children) ||
      push_key_handler<ArrayHandler<int>>("parents", parents) ||
      push_key_handler<ArrayHandler<int>>("split_indices", split_indices) ||
      push_key_handler<ArrayHandler<int>>("split_type", split_type) ||
      push_key_handler<ArrayHandler<double>>("split_conditions", split_conditions) ||
      push_key_handler<ArrayHandler<bool>>("default_left", default_left));
}

bool RegTreeHandler::StartObject() {
  return push_key_handler<TreeParamHandler, int>("tree_param", num_nodes);
}

bool RegTreeHandler::Uint(unsigned) { return check_cur_key("id"); }

bool RegTreeHandler::EndObject(std::size_t) {
  output.Init();
  if (split_type.empty()) {
    split_type.resize(num_nodes, details::xgboost::FeatureType::kNumerical);
  }
  if (static_cast<size_t>(num_nodes) != loss_changes.size()) {
    TREELITE_LOG(ERROR) << "Field loss_changes has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << loss_changes.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != sum_hessian.size()) {
    TREELITE_LOG(ERROR) << "Field sum_hessian has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << sum_hessian.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != base_weights.size()) {
    TREELITE_LOG(ERROR) << "Field base_weights has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << base_weights.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != left_children.size()) {
    TREELITE_LOG(ERROR) << "Field left_children has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << left_children.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != right_children.size()) {
    TREELITE_LOG(ERROR) << "Field right_children has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << right_children.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != parents.size()) {
    TREELITE_LOG(ERROR) << "Field parents has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << parents.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != split_indices.size()) {
    TREELITE_LOG(ERROR) << "Field split_indices has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << split_indices.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != split_type.size()) {
    TREELITE_LOG(ERROR) << "Field split_type has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << split_type.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != split_conditions.size()) {
    TREELITE_LOG(ERROR) << "Field split_conditions has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << split_conditions.size();
    return false;
  }
  if (static_cast<size_t>(num_nodes) != default_left.size()) {
    TREELITE_LOG(ERROR) << "Field default_left has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << default_left.size();
    return false;
  }

  std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
  if (num_nodes > 0) {
    Q.push({0, 0});
  }
  while (!Q.empty()) {
    int old_id, new_id;
    std::tie(old_id, new_id) = Q.front();
    Q.pop();

    if (left_children[old_id] == -1) {
      output.SetLeaf(new_id, split_conditions[old_id]);
    } else {
      output.AddChilds(new_id);
      if (split_type[old_id] == details::xgboost::FeatureType::kCategorical) {
        auto categorical_split_loc
          = math::binary_search(categories_nodes.begin(), categories_nodes.end(), old_id);
        TREELITE_CHECK(categorical_split_loc != categories_nodes.end())
          << "Could not find record for the categorical split in node " << old_id;
        auto categorical_split_id = std::distance(categories_nodes.begin(), categorical_split_loc);
        int offset = categories_segments[categorical_split_id];
        int num_categories = categories_sizes[categorical_split_id];
        std::vector<uint32_t> right_categories;
        right_categories.reserve(num_categories);
        for (int i = 0; i < num_categories; ++i) {
          right_categories.push_back(static_cast<uint32_t>(categories[offset + i]));
        }
        output.SetCategoricalSplit(
            new_id, split_indices[old_id], default_left[old_id], right_categories, true);
      } else {
        output.SetNumericalSplit(
            new_id, split_indices[old_id], split_conditions[old_id],
            default_left[old_id], treelite::Operator::kLT);
      }
      output.SetGain(new_id, loss_changes[old_id]);
      Q.push({left_children[old_id], output.LeftChild(new_id)});
      Q.push({right_children[old_id], output.RightChild(new_id)});
    }
    output.SetSumHess(new_id, sum_hessian[old_id]);
  }
  return pop_handler();
}

/******************************************************************************
 * GBTreeHandler
 * ***************************************************************************/
bool GBTreeModelHandler::StartArray() {
  return (push_key_handler<ArrayHandler<treelite::Tree<float, float>, RegTreeHandler>,
                           std::vector<treelite::Tree<float, float>>>(
                               "trees", output.trees) ||
          push_key_handler<IgnoreHandler>("tree_info"));
}

bool GBTreeModelHandler::StartObject() {
  return push_key_handler<IgnoreHandler>("gbtree_model_param");
}

/******************************************************************************
 * GradientBoosterHandler
 * ***************************************************************************/
bool GradientBoosterHandler::String(const char *str,
                                    std::size_t length,
                                    bool) {
  if (assign_value("name", std::string{str, length}, name)) {
    if (name == "gbtree" || name == "dart") {
      return true;
    } else {
      TREELITE_LOG(ERROR) << "Only GBTree or DART boosters are currently supported.";
      return false;
    }
  } else {
    return false;
  }
}
bool GradientBoosterHandler::StartObject() {
  if (push_key_handler<GBTreeModelHandler, treelite::ModelImpl<float, float>>("model", output)) {
    return true;
  } else if (push_key_handler<GradientBoosterHandler, treelite::ModelImpl<float, float>>("gbtree",
                                                                                         output)) {
    // "dart" booster contains a standard gbtree under ["gradient_booster"]["gbtree"]["model"].
    return true;
  } else {
    TREELITE_LOG(ERROR) << "Key \"" << get_cur_key()
                        << "\" not recognized. Is this a GBTree-type booster?";
    return false;
  }
}
bool GradientBoosterHandler::StartArray() {
  return push_key_handler<ArrayHandler<double>, std::vector<double>>("weight_drop", weight_drop);
}
bool GradientBoosterHandler::EndObject(std::size_t memberCount) {
  if (name == "dart" && !weight_drop.empty()) {
    // Fold weight drop into leaf value for dart models.
    TREELITE_CHECK_EQ(output.trees.size(), weight_drop.size());
    for (size_t i = 0; i < output.trees.size(); ++i) {
      for (int nid = 0; nid < output.trees[i].num_nodes; ++nid) {
        if (output.trees[i].IsLeaf(nid)) {
          output.trees[i].SetLeaf(nid, weight_drop[i] * output.trees[i].LeafValue(nid));
        }
      }
    }
  }
  return pop_handler();
}

/******************************************************************************
 * ObjectiveHandler
 * ***************************************************************************/
bool ObjectiveHandler::StartObject() {
  return (push_key_handler<IgnoreHandler>("reg_loss_param") ||
          push_key_handler<IgnoreHandler>("poisson_regression_param") ||
          push_key_handler<IgnoreHandler>("tweedie_regression_param") ||
          push_key_handler<IgnoreHandler>("softmax_multiclass_param") ||
          push_key_handler<IgnoreHandler>("lambda_rank_param") ||
          push_key_handler<IgnoreHandler>("aft_loss_param"));
}

bool ObjectiveHandler::String(const char *str, std::size_t length, bool) {
  return assign_value("name", std::string{str, length}, output);
}

/******************************************************************************
 * LearnerParamHandler
 * ***************************************************************************/
bool LearnerParamHandler::String(const char *str,
                                 std::size_t,
                                 bool) {
  return (assign_value("base_score", strtof(str, nullptr),
                       output.param.global_bias) ||
          assign_value("num_class", static_cast<unsigned int>(std::max(std::atoi(str), 1)),
                       output.task_param.num_class) ||
          assign_value("num_feature", std::atoi(str), output.num_feature));
}

/******************************************************************************
 * LearnerHandler
 * ***************************************************************************/
bool LearnerHandler::StartObject() {
  // "attributes" key is not documented in schema
  return (push_key_handler<LearnerParamHandler, treelite::ModelImpl<float, float>>(
              "learner_model_param", *output.model) ||
          push_key_handler<GradientBoosterHandler, treelite::ModelImpl<float, float>>(
              "gradient_booster", *output.model) ||
          push_key_handler<ObjectiveHandler, std::string>("objective", objective) ||
          push_key_handler<IgnoreHandler>("attributes"));
}

bool LearnerHandler::EndObject(std::size_t) {
  xgboost::SetPredTransform(objective, &output.model->param);
  output.objective_name = objective;
  return pop_handler();
}

bool LearnerHandler::StartArray() {
  return (push_key_handler<IgnoreHandler>("feature_names") ||
          push_key_handler<IgnoreHandler>("feature_types"));
}

/******************************************************************************
 * XGBoostModelHandler
 * ***************************************************************************/
bool XGBoostModelHandler::StartArray() {
  return push_key_handler<ArrayHandler<unsigned>, std::vector<unsigned>>(
      "version", version);
}

bool XGBoostModelHandler::StartObject() {
  return push_key_handler<LearnerHandler, XGBoostModelHandle>("learner", output);
}

bool XGBoostModelHandler::EndObject(std::size_t memberCount) {
  if (memberCount != 2) {
    TREELITE_LOG(ERROR) << "Expected two members in XGBoostModel";
    return false;
  }
  output.model->average_tree_output = false;
  output.model->task_param.output_type = TaskParam::OutputType::kFloat;
  output.model->task_param.leaf_vector_size = 1;
  if (output.model->task_param.num_class > 1) {
    // multi-class classifier
    output.model->task_type = TaskType::kMultiClfGrovePerClass;
    output.model->task_param.grove_per_class = true;
  } else {
    // binary classifier or regressor
    output.model->task_type = TaskType::kBinaryClfRegr;
    output.model->task_param.grove_per_class = false;
  }
  // Before XGBoost 1.0.0, the global bias saved in model is a transformed value.  After
  // 1.0 it's the original value provided by user.
  const bool need_transform_to_margin = (version[0] >= 1);
  if (need_transform_to_margin) {
    treelite::details::xgboost::TransformGlobalBiasToMargin(&output.model->param);
  }
  return pop_handler();
}

/******************************************************************************
 * RootHandler
 * ***************************************************************************/
bool RootHandler::StartObject() {
  handle = {dynamic_cast<treelite::ModelImpl<float, float>*>(output.get()), ""};
  return push_handler<XGBoostModelHandler, XGBoostModelHandle>(handle);
}

/******************************************************************************
 * DelegatedHandler
 * ***************************************************************************/
std::unique_ptr<treelite::Model> DelegatedHandler::get_result() { return std::move(result); }
bool DelegatedHandler::Null() { return delegates.top()->Null(); }
bool DelegatedHandler::Bool(bool b) { return delegates.top()->Bool(b); }
bool DelegatedHandler::Int(int i) { return delegates.top()->Int(i); }
bool DelegatedHandler::Uint(unsigned u) { return delegates.top()->Uint(u); }
bool DelegatedHandler::Int64(int64_t i) { return delegates.top()->Int64(i); }
bool DelegatedHandler::Uint64(uint64_t u) { return delegates.top()->Uint64(u); }
bool DelegatedHandler::Double(double d) { return delegates.top()->Double(d); }
bool DelegatedHandler::String(const char *str, std::size_t length, bool copy) {
  return delegates.top()->String(str, length, copy);
}
bool DelegatedHandler::StartObject() { return delegates.top()->StartObject(); }
bool DelegatedHandler::Key(const char *str, std::size_t length, bool copy) {
  return delegates.top()->Key(str, length, copy);
}
bool DelegatedHandler::EndObject(std::size_t memberCount) {
  return delegates.top()->EndObject(memberCount);
}
bool DelegatedHandler::StartArray() { return delegates.top()->StartArray(); }
bool DelegatedHandler::EndArray(std::size_t elementCount) {
  return delegates.top()->EndArray(elementCount);
}


}  // namespace details
}  // namespace treelite

namespace {
template <typename StreamType, typename ErrorHandlerFunc>
std::unique_ptr<treelite::Model> ParseStream(std::unique_ptr<StreamType> input_stream,
                                             ErrorHandlerFunc error_handler) {
  std::shared_ptr<treelite::details::DelegatedHandler> handler =
    treelite::details::DelegatedHandler::create();
  rapidjson::Reader reader;

  rapidjson::ParseResult result
    = reader.Parse<rapidjson::ParseFlag::kParseNanAndInfFlag>(*input_stream, *handler);
  if (!result) {
    const auto error_code = result.Code();
    const size_t offset = result.Offset();
    std::string diagnostic = error_handler(offset);
    TREELITE_LOG(FATAL) << "Provided JSON could not be parsed as XGBoost model. Parsing error at offset "
                        << offset << ": " << rapidjson::GetParseError_En(error_code) << "\n"
                        << diagnostic;
  }
  return handler->get_result();
}
}  // anonymous namespace
