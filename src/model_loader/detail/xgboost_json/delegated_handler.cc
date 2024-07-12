/*!
 * Copyright (c) 2020-2024 by Contributors
 * \file delegated_handler.cc
 * \brief A delegated handler for loading XGBoost JSON/UBJSON models via SAX parsers.
 *        The delegated handler delegates the parsing of components to sub-handlers.
 * \author Hyunsu Cho, William Hicks
 */

#include "./delegated_handler.h"

#include <string>

#include <treelite/logging.h>

#include "../string_utils.h"
#include "../xgboost.h"

namespace {

/*!
 * \brief Perform binary search on the range [begin, end).
 * \param begin Beginning of the search range
 * \param end End of the search range
 * \param val Value being searched
 * \return Iterator pointing to the value if found; end if value not found.
 * \tparam Iter Type of iterator
 * \tparam T Type of elements
 */
template <class Iter, class T>
Iter BinarySearch(Iter begin, Iter end, T const& val) {
  Iter i = std::lower_bound(begin, end, val);
  if (i != end && val == *i) {
    return i;  // found
  } else {
    return end;  // not found
  }
}

}  // anonymous namespace

namespace treelite::model_loader::detail::xgboost {

class IgnoreHandler;

/******************************************************************************
 * BaseHandler
 ******************************************************************************/

BaseHandler::BaseHandler(std::weak_ptr<Delegator> parent_delegator)
    : delegator{parent_delegator}, allow_unknown_field_{false}, state_next_field_ignore_{false} {
  if (auto parent = delegator.lock()) {
    // Configure the allow_unknown_field flag
    auto const& handler_config = parent->get_handler_config();
    allow_unknown_field_ = handler_config.allow_unknown_field;
  }
}

bool BaseHandler::Null() {
  if (should_ignore_upcoming_value()) {
    return true;
  }
  TREELITE_LOG(ERROR)
      << "Reached a dummy handler BaseHandler::Null(). There is likely a bug in the model parser.";
  return false;
}

bool BaseHandler::Bool(bool) {
  if (should_ignore_upcoming_value()) {
    return true;
  }
  TREELITE_LOG(ERROR)
      << "Reached a dummy handler BaseHandler::Bool(). There is likely a bug in the model parser.";
  return false;
}

bool BaseHandler::Int64(std::int64_t) {
  if (should_ignore_upcoming_value()) {
    return true;
  }
  TREELITE_LOG(ERROR)
      << "Reached a dummy handler BaseHandler::Int64(). There is likely a bug in the model parser.";
  return false;
}

bool BaseHandler::Uint64(std::uint64_t) {
  if (should_ignore_upcoming_value()) {
    return true;
  }
  TREELITE_LOG(ERROR) << "Reached a dummy handler BaseHandler::Uint64(). "
                      << "There is likely a bug in the model parser.";
  return false;
}

bool BaseHandler::Double(double) {
  if (should_ignore_upcoming_value()) {
    return true;
  }
  TREELITE_LOG(ERROR) << "Reached a dummy handler BaseHandler::Double(). "
                      << "There is likely a bug in the model parser.";
  return false;
}

bool BaseHandler::String(std::string const&) {
  if (should_ignore_upcoming_value()) {
    return true;
  }
  TREELITE_LOG(ERROR) << "Reached a dummy handler BaseHandler::String(). "
                      << "There is likely a bug in the model parser.";
  return false;
}

bool BaseHandler::StartObject() {
  if (should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  TREELITE_LOG(ERROR) << "Reached a dummy handler BaseHandler::StartObject(). "
                      << "There is likely a bug in the model parser.";
  return false;
}

bool BaseHandler::Key(std::string const& str) {
  return set_cur_key(str);
}

bool BaseHandler::EndObject() {
  return pop_handler();
}

bool BaseHandler::StartArray() {
  if (should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  TREELITE_LOG(ERROR) << "Reached a dummy handler BaseHandler::StartArray(). "
                      << "There is likely a bug in the model parser.";
  return false;
}

bool BaseHandler::EndArray() {
  return pop_handler();
}

bool BaseHandler::pop_handler() {
  if (auto parent = delegator.lock()) {
    parent->pop_delegate();
    return true;
  } else {
    return false;
  }
}

bool BaseHandler::set_cur_key(std::string const& key) {
  if (is_recognized_key(key)) {
    cur_key = key;
  } else if (allow_unknown_field_) {
    TREELITE_LOG(WARNING) << "Warning: Encountered unknown key \"" << key << "\"";
    cur_key = "";
    state_next_field_ignore_ = true;
  } else {
    // Extra field with unknown key is a fatal error if allow_unknown_field_ is not set
    TREELITE_LOG(ERROR) << "Error: key \"" << key << "\" is not recognized!";
    return false;
  }
  return true;
}

std::string const& BaseHandler::get_cur_key() {
  return cur_key;
}

bool BaseHandler::check_cur_key(std::string const& query_key) {
  return cur_key == query_key;
}

template <typename ValueType>
bool BaseHandler::assign_value(std::string const& key, ValueType&& value, ValueType& output) {
  if (check_cur_key(key)) {
    output = value;
    return true;
  } else {
    return false;
  }
}

template <typename ValueType>
bool BaseHandler::assign_value(std::string const& key, ValueType const& value, ValueType& output) {
  if (check_cur_key(key)) {
    output = value;
    return true;
  } else {
    return false;
  }
}

/******************************************************************************
 * DelegatedHandler
 * ***************************************************************************/

DelegatedHandler::DelegatedHandler(HandlerConfig const& handler_config)
    : delegates{},
      result{model_builder::GetModelBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32), 0, {}, {}, ""},
      handler_config_{handler_config} {}

std::shared_ptr<DelegatedHandler> DelegatedHandler::create_empty(
    HandlerConfig const& handler_config) {
  struct make_shared_enabler : public DelegatedHandler {
    explicit make_shared_enabler(HandlerConfig const& handler_config)
        : DelegatedHandler{handler_config} {}
  };
  std::shared_ptr<DelegatedHandler> new_handler
      = std::make_shared<make_shared_enabler>(handler_config);
  return new_handler;
}

std::shared_ptr<DelegatedHandler> DelegatedHandler::create(HandlerConfig const& handler_config) {
  std::shared_ptr<DelegatedHandler> new_handler = create_empty(handler_config);
  new_handler->push_delegate(std::make_shared<RootHandler>(new_handler, new_handler->result));
  return new_handler;
}

void DelegatedHandler::push_delegate(std::shared_ptr<BaseHandler> new_delegate) {
  delegates.push(new_delegate);
}

void DelegatedHandler::pop_delegate() {
  delegates.pop();
}

ParsedXGBoostModel DelegatedHandler::get_result() {
  return std::move(result);
}

bool DelegatedHandler::Null() {
  return delegates.top()->Null();
}

bool DelegatedHandler::Bool(bool b) {
  return delegates.top()->Bool(b);
}

bool DelegatedHandler::Int64(int64_t i) {
  return delegates.top()->Int64(i);
}

bool DelegatedHandler::Uint64(uint64_t u) {
  return delegates.top()->Uint64(u);
}

bool DelegatedHandler::Double(double d) {
  return delegates.top()->Double(d);
}

bool DelegatedHandler::String(std::string const& str) {
  return delegates.top()->String(str);
}

bool DelegatedHandler::StartObject() {
  return delegates.top()->StartObject();
}

bool DelegatedHandler::Key(std::string const& key) {
  return delegates.top()->Key(key);
}

bool DelegatedHandler::EndObject() {
  return delegates.top()->EndObject();
}

bool DelegatedHandler::StartArray() {
  return delegates.top()->StartArray();
}

bool DelegatedHandler::EndArray() {
  return delegates.top()->EndArray();
}

/******************************************************************************
 * IgnoreHandler
 ******************************************************************************/

bool IgnoreHandler::Null() {
  return true;
}

bool IgnoreHandler::Bool(bool) {
  return true;
}

bool IgnoreHandler::Int64(int64_t) {
  return true;
}

bool IgnoreHandler::Uint64(uint64_t) {
  return true;
}

bool IgnoreHandler::Double(double) {
  return true;
}

bool IgnoreHandler::String(std::string const&) {
  return true;
}

bool IgnoreHandler::StartObject() {
  return push_handler<IgnoreHandler>();
}

bool IgnoreHandler::Key(std::string const&) {
  return true;
}

bool IgnoreHandler::StartArray() {
  return push_handler<IgnoreHandler>();
}

/******************************************************************************
 * TreeParamHandler
 * ***************************************************************************/
bool TreeParamHandler::String(std::string const& str) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  // Key "num_deleted" deprecated but still present in some xgboost output
  return (check_cur_key("num_feature")
          || assign_value("num_nodes", std::stoi(str), output.num_nodes)
          || assign_value("size_leaf_vector", std::stoi(str), output.size_leaf_vector)
          || check_cur_key("num_deleted"));
}

bool TreeParamHandler::is_recognized_key(std::string const& key) {
  return (key == "num_feature" || key == "num_nodes" || key == "size_leaf_vector"
          || key == "num_deleted");
}

/******************************************************************************
 * RegTreeHandler
 * ***************************************************************************/

RegTreeHandler::RegTreeHandler(std::weak_ptr<Delegator> parent_delegator,
    ParsedRegTreeParams& output, model_builder::ModelBuilder& model_builder)
    : OutputHandler{parent_delegator, output}, model_builder{model_builder} {}

bool RegTreeHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return (push_key_handler<ArrayHandler<float>>("loss_changes", loss_changes)
          || push_key_handler<ArrayHandler<float>>("sum_hessian", sum_hessian)
          || push_key_handler<ArrayHandler<float>>("base_weights", base_weights)
          || push_key_handler<ArrayHandler<int>>("categories_segments", categories_segments)
          || push_key_handler<ArrayHandler<int>>("categories_sizes", categories_sizes)
          || push_key_handler<ArrayHandler<int>>("categories_nodes", categories_nodes)
          || push_key_handler<ArrayHandler<int>>("categories", categories)
          || push_key_handler<IgnoreHandler>("leaf_child_counts")
          || push_key_handler<ArrayHandler<int>>("left_children", left_children)
          || push_key_handler<ArrayHandler<int>>("right_children", right_children)
          || push_key_handler<ArrayHandler<int>>("parents", parents)
          || push_key_handler<ArrayHandler<int>>("split_indices", split_indices)
          || push_key_handler<ArrayHandler<int>>("split_type", split_type)
          || push_key_handler<ArrayHandler<float>>("split_conditions", split_conditions)
          || push_key_handler<ArrayHandler<bool>>("default_left", default_left));
}

bool RegTreeHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<TreeParamHandler, ParsedRegTreeParams>("tree_param", output);
}

bool RegTreeHandler::Int64(std::int64_t x) {
  // The "id" field can be int64 or uint64
  // Just defer to the uint64 handler
  return Uint64(static_cast<std::uint64_t>(x));
}

bool RegTreeHandler::Uint64(std::uint64_t) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  return check_cur_key("id");
}

bool RegTreeHandler::EndObject() {
  auto const num_nodes = output.num_nodes;
  if (split_type.empty()) {
    split_type.resize(num_nodes, xgboost::FeatureType::kNumerical);
  }
  if (output.size_leaf_vector == 0) {
    output.size_leaf_vector = 1;  // In XGBoost, size_leaf_vector=0 indicates a scalar output
  }
  if (num_nodes * output.size_leaf_vector != base_weights.size()) {
    TREELITE_LOG(ERROR) << "Field base_weights has an incorrect dimension. Expected: "
                        << (num_nodes * output.size_leaf_vector)
                        << ", Actual: " << base_weights.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != left_children.size()) {
    TREELITE_LOG(ERROR) << "Field left_children has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << left_children.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != right_children.size()) {
    TREELITE_LOG(ERROR) << "Field right_children has an incorrect dimension. Expected: "
                        << num_nodes << ", Actual: " << right_children.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != parents.size()) {
    TREELITE_LOG(ERROR) << "Field parents has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << parents.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != split_indices.size()) {
    TREELITE_LOG(ERROR) << "Field split_indices has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << split_indices.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != split_type.size()) {
    TREELITE_LOG(ERROR) << "Field split_type has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << split_type.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != split_conditions.size()) {
    TREELITE_LOG(ERROR) << "Field split_conditions has an incorrect dimension. Expected: "
                        << num_nodes << ", Actual: " << split_conditions.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != default_left.size()) {
    TREELITE_LOG(ERROR) << "Field default_left has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << default_left.size();
    return false;
  }

  model_builder.StartTree();
  for (std::int32_t node_id = 0; node_id < num_nodes; ++node_id) {
    model_builder.StartNode(node_id);
    if (left_children[node_id] == -1) {
      auto const size_leaf_vector = output.size_leaf_vector;
      if (size_leaf_vector > 1) {
        // Vector output
        std::vector<float> leafvec(size_leaf_vector);
        std::transform(&base_weights[node_id * size_leaf_vector],
            &base_weights[(node_id + 1) * size_leaf_vector], leafvec.begin(),
            [](float e) { return static_cast<float>(e); });
        model_builder.LeafVector(leafvec);
      } else {
        // Scalar leaf output
        model_builder.LeafScalar(split_conditions[node_id]);
      }
    } else {
      if (split_type[node_id] == xgboost::FeatureType::kCategorical) {
        auto categorical_split_loc
            = BinarySearch(categories_nodes.begin(), categories_nodes.end(), node_id);
        TREELITE_CHECK(categorical_split_loc != categories_nodes.end())
            << "Could not find record for the categorical split in node " << node_id;
        auto categorical_split_id = std::distance(categories_nodes.begin(), categorical_split_loc);
        int offset = categories_segments[categorical_split_id];
        int num_categories = categories_sizes[categorical_split_id];
        std::vector<std::uint32_t> right_categories;
        right_categories.reserve(num_categories);
        for (int i = 0; i < num_categories; ++i) {
          right_categories.push_back(static_cast<std::uint32_t>(categories[offset + i]));
        }
        model_builder.CategoricalTest(split_indices[node_id], default_left[node_id],
            right_categories, true, left_children[node_id], right_children[node_id]);
      } else {
        model_builder.NumericalTest(split_indices[node_id], split_conditions[node_id],
            default_left[node_id], Operator::kLT, left_children[node_id], right_children[node_id]);
      }
      if (!loss_changes.empty()) {
        model_builder.Gain(loss_changes[node_id]);
      }
    }
    if (!sum_hessian.empty()) {
      model_builder.SumHess(sum_hessian[node_id]);
    }
    model_builder.EndNode();
  }
  model_builder.EndTree();
  return pop_handler();
}

bool RegTreeHandler::is_recognized_key(std::string const& key) {
  return (key == "loss_changes" || key == "sum_hessian" || key == "base_weights"
          || key == "categories_segments" || key == "categories_sizes" || key == "categories_nodes"
          || key == "categories" || key == "leaf_child_counts" || key == "left_children"
          || key == "right_children" || key == "parents" || key == "split_indices"
          || key == "split_type" || key == "split_conditions" || key == "default_left"
          || key == "tree_param" || key == "id");
}

/******************************************************************************
 * RegTreeArrayHandler
 * ***************************************************************************/

RegTreeArrayHandler::RegTreeArrayHandler(std::weak_ptr<Delegator> parent_delegator,
    std::vector<ParsedRegTreeParams>& output, model_builder::ModelBuilder& model_builder)
    : OutputHandler{parent_delegator, output}, model_builder{model_builder} {}

bool RegTreeArrayHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return this->template push_handler<IgnoreHandler>();
  }
  this->output.emplace_back();
  return this->template push_handler<RegTreeHandler, ParsedRegTreeParams>(
      this->output.back(), model_builder);
}

/******************************************************************************
 * GBTreeHandler
 * ***************************************************************************/

bool GBTreeModelHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  output.builder->SetValidationFlag("check_orphaned_nodes", false);
  return (push_key_handler<RegTreeArrayHandler, std::vector<ParsedRegTreeParams>>(
              "trees", reg_tree_params, *output.builder)
          || push_key_handler<ArrayHandler<int>, std::vector<int>>("tree_info", output.tree_info)
          || push_key_handler<IgnoreHandler>("iteration_indptr"));
}

bool GBTreeModelHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<IgnoreHandler>("gbtree_model_param");
}

bool GBTreeModelHandler::EndObject() {
  if (!reg_tree_params.empty()) {
    TREELITE_CHECK_LE(
        reg_tree_params.size(), static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        << "Too many trees";
    output.num_tree = static_cast<std::int32_t>(reg_tree_params.size());
    output.size_leaf_vector = reg_tree_params[0].size_leaf_vector;
    for (ParsedRegTreeParams const& e : reg_tree_params) {
      TREELITE_CHECK_EQ(e.size_leaf_vector, output.size_leaf_vector)
          << "We currently don't support loading model whose trees have different output size";
    }
  }
  return pop_handler();
}

bool GBTreeModelHandler::is_recognized_key(std::string const& key) {
  return (key == "trees" || key == "tree_info" || key == "gbtree_model_param"
          || key == "iteration_indptr");
}

/******************************************************************************
 * GradientBoosterHandler
 * ***************************************************************************/

bool GradientBoosterHandler::String(std::string const& str) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  if (assign_value("name", str, name)) {
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
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  if (push_key_handler<GBTreeModelHandler, ParsedXGBoostModel>("model", output)) {
    return true;
  } else if (push_key_handler<GradientBoosterHandler, ParsedXGBoostModel>("gbtree", output)) {
    // "dart" booster contains a standard gbtree under ["gradient_booster"]["gbtree"]["model"].
    return true;
  } else {
    TREELITE_LOG(ERROR) << "Key \"" << get_cur_key()
                        << "\" not recognized. Is this a GBTree-type booster?";
    return false;
  }
}

bool GradientBoosterHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<ArrayHandler<float>, std::vector<float>>("weight_drop", weight_drop);
}

bool GradientBoosterHandler::EndObject() {
  if (name == "dart" && !weight_drop.empty()) {
    TREELITE_CHECK_EQ(output.size_leaf_vector, 1)
        << "Dart with vector-leaf output is not yet supported";
    output.weight_drop = weight_drop;
  }
  return pop_handler();
}

bool GradientBoosterHandler::is_recognized_key(std::string const& key) {
  return (key == "name" || key == "model" || key == "gbtree" || key == "weight_drop");
}

/******************************************************************************
 * ObjectiveHandler
 * ***************************************************************************/

bool ObjectiveHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  // pseduo_huber_param is a mispelled variant of pseudo_huber_param; it's used in a past version
  // of XGBoost
  return (push_key_handler<IgnoreHandler>("reg_loss_param")
          || push_key_handler<IgnoreHandler>("poisson_regression_param")
          || push_key_handler<IgnoreHandler>("tweedie_regression_param")
          || push_key_handler<IgnoreHandler>("softmax_multiclass_param")
          || push_key_handler<IgnoreHandler>("lambda_rank_param")
          || push_key_handler<IgnoreHandler>("aft_loss_param")
          || push_key_handler<IgnoreHandler>("pseduo_huber_param")
          || push_key_handler<IgnoreHandler>("pseudo_huber_param")
          || push_key_handler<IgnoreHandler>("lambdarank_param"));
}

bool ObjectiveHandler::String(std::string const& str) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  return assign_value("name", str, output);
}

bool ObjectiveHandler::is_recognized_key(std::string const& key) {
  return (key == "reg_loss_param" || key == "poisson_regression_param"
          || key == "tweedie_regression_param" || key == "softmax_multiclass_param"
          || key == "lambda_rank_param" || key == "aft_loss_param" || key == "pseduo_huber_param"
          || key == "pseudo_huber_param" || key == "lambdarank_param" || key == "name");
}

/******************************************************************************
 * LearnerParamHandler
 * ***************************************************************************/

bool LearnerParamHandler::String(std::string const& str) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  // For now, XGBoost always outputs a scalar base_score
  return (
      assign_value("base_score", static_cast<float>(std::stof(str)), output.base_score)
      || assign_value("num_class", std::max(std::stoi(str), 1), output.num_class)
      || assign_value("num_target", static_cast<std::int32_t>(std::stoi(str)), output.num_target)
      || assign_value("num_feature", std::stoi(str), output.num_feature)
      || assign_value(
          "boost_from_average", static_cast<bool>(std::stoi(str)), output.boost_from_average));
}

bool LearnerParamHandler::is_recognized_key(std::string const& key) {
  return (key == "num_target" || key == "base_score" || key == "num_class" || key == "num_feature"
          || key == "boost_from_average");
}

/******************************************************************************
 * LearnerHandler
 * ***************************************************************************/

bool LearnerHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  // "attributes" key is not documented in schema
  return (
      push_key_handler<LearnerParamHandler, ParsedLearnerParams>(
          "learner_model_param", learner_params)
      || push_key_handler<GradientBoosterHandler, ParsedXGBoostModel>("gradient_booster", output)
      || push_key_handler<ObjectiveHandler, std::string>("objective", objective)
      || push_key_handler<IgnoreHandler>("attributes"));
}

bool LearnerHandler::EndObject() {
  /* Set metadata */
  auto const num_tree = output.num_tree;
  auto const num_feature = learner_params.num_feature;
  bool const average_tree_output = false;
  auto const num_target = learner_params.num_target;

  model_builder::PostProcessorFunc postprocessor{xgboost::GetPostProcessor(objective)};
  output.objective_name = objective;  // Save objective name, to use later

  TaskType task_type;
  std::vector<std::int32_t> num_class;
  std::vector<std::int32_t> target_id, class_id;
  std::array<std::int32_t, 2> leaf_vector_shape{1, 1};
  if (learner_params.num_class > 1) {  // Multi-class classifier
    // For now, XGBoost does not support multi-target models for multi-class classification
    // So if num_class > 1, we can assume num_target == 1
    TREELITE_CHECK_EQ(learner_params.num_target, 1)
        << "XGBoost does not support multi-target models for multi-class classification";
    num_class = std::vector<std::int32_t>{learner_params.num_class};
    task_type = TaskType::kMultiClf;
    target_id = std::vector<std::int32_t>(num_tree, 0);
    TREELITE_CHECK_GT(output.size_leaf_vector, 0);
    if (output.size_leaf_vector > 1) {
      // Vector-leaf output
      class_id = std::vector<std::int32_t>(num_tree, -1);
    } else {
      // Grove per class: i-th tree produces output for class (i % num_class)
      // Note: num_parallel_tree can change this behavior, so it's best to go with
      // tree_info field provided by XGBoost
      class_id = std::vector<std::int32_t>(num_tree);
      for (std::int32_t tree_id = 0; tree_id < num_tree; ++tree_id) {
        class_id[tree_id] = static_cast<std::int32_t>(output.tree_info[tree_id]);
      }
    }
    leaf_vector_shape[0] = 1;
    leaf_vector_shape[1] = output.size_leaf_vector;
  } else {
    // Binary classifier or regressor
    if (StringStartsWith(output.objective_name, "binary:")) {
      task_type = TaskType::kBinaryClf;
    } else if (StringStartsWith(output.objective_name, "rank:")) {
      task_type = TaskType::kLearningToRank;
    } else {
      task_type = TaskType::kRegressor;
    }
    num_class = std::vector<std::int32_t>(num_target, 1);
    class_id = std::vector<std::int32_t>(num_tree, 0);
    if (output.size_leaf_vector > 1) {
      // Vector-leaf output
      target_id = std::vector<std::int32_t>(num_tree, -1);
      TREELITE_CHECK_EQ(num_target, static_cast<std::int32_t>(output.size_leaf_vector));
      leaf_vector_shape[0] = num_target;
      leaf_vector_shape[1] = 1;
    } else {
      // Grove per target: i-th tree produces output for target (i % num_target)
      target_id = std::vector<std::int32_t>(num_tree);
      for (std::int32_t tree_id = 0; tree_id < num_tree; ++tree_id) {
        target_id[tree_id] = static_cast<std::int32_t>(output.tree_info[tree_id]);
      }
      leaf_vector_shape[0] = 1;
      leaf_vector_shape[1] = 1;
    }
  }
  // Set base scores. For now, XGBoost only supports a scalar base score for all targets / classes.
  auto base_score = static_cast<double>(learner_params.base_score);
  // Before XGBoost 1.0.0, the base score saved in model is a transformed value.  After
  // 1.0 it's the original value provided by user.
  bool const need_transform_to_margin = output.version.empty() || output.version[0] >= 1;
  if (need_transform_to_margin) {
    base_score = xgboost::TransformBaseScoreToMargin(postprocessor.name, base_score);
  }
  // For now, XGBoost produces a scalar base_score
  // Assume: Either num_target or num_class must be 1
  TREELITE_CHECK(learner_params.num_target == 1 || learner_params.num_class == 1);
  std::vector<double> base_scores(learner_params.num_target * learner_params.num_class, base_score);

  model_builder::Metadata metadata{
      num_feature, task_type, average_tree_output, num_target, num_class, leaf_vector_shape};
  model_builder::TreeAnnotation tree_annotation{num_tree, target_id, class_id};
  output.builder->InitializeMetadata(
      metadata, tree_annotation, postprocessor, base_scores, std::nullopt);

  return pop_handler();
}

bool LearnerHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return (push_key_handler<IgnoreHandler>("feature_names")
          || push_key_handler<IgnoreHandler>("feature_types"));
}

bool LearnerHandler::is_recognized_key(std::string const& key) {
  return (key == "learner_model_param" || key == "gradient_booster" || key == "objective"
          || key == "attributes" || key == "feature_names" || key == "feature_types");
}

/******************************************************************************
 * XGBoostCheckpointHandler
 * ***************************************************************************/

bool XGBoostCheckpointHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<ArrayHandler<unsigned>, std::vector<unsigned>>("version", output.version);
}

bool XGBoostCheckpointHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<LearnerHandler, ParsedXGBoostModel>("learner", output);
}

bool XGBoostCheckpointHandler::is_recognized_key(std::string const& key) {
  return (key == "version" || key == "learner");
}

/******************************************************************************
 * XGBoostModelHandler
 * ***************************************************************************/

bool XGBoostModelHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<ArrayHandler<unsigned>, std::vector<unsigned>>("version", output.version);
}

bool XGBoostModelHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return (push_key_handler<LearnerHandler, ParsedXGBoostModel>("learner", output)
          || push_key_handler<IgnoreHandler>("Config")
          || push_key_handler<XGBoostCheckpointHandler, ParsedXGBoostModel>("Model", output));
}

bool XGBoostModelHandler::is_recognized_key(std::string const& key) {
  return (key == "version" || key == "learner" || key == "Config" || key == "Model");
}

/******************************************************************************
 * RootHandler
 * ***************************************************************************/

bool RootHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_handler<XGBoostModelHandler, ParsedXGBoostModel>(output);
}

}  // namespace treelite::model_loader::detail::xgboost
