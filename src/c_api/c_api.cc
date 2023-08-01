/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file c_api.cc
 * \author Hyunsu Cho
 * \brief C API of Treelite, used for interfacing with other languages
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/data.h>
#include <treelite/frontend.h>
#include <treelite/gtil.h>
#include <treelite/logging.h>
#include <treelite/math.h>
#include <treelite/thread_local.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

using namespace treelite;  // NOLINT(build/namespaces)

namespace {

/*! \brief entry to to easily hold returning information */
struct TreeliteAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief Temp variable for returning prediction shape. */
  std::vector<std::size_t> prediction_shape;
};

// define threadlocal store for returning information
using TreeliteAPIThreadLocalStore = ThreadLocalStore<TreeliteAPIThreadLocalEntry>;

}  // anonymous namespace

int TreeliteLoadLightGBMModel(char const* filename, ModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadLightGBMModel() is deprecated. Please use "
                        << "TreeliteLoadLightGBMModelEx() instead.";
  return TreeliteLoadLightGBMModelEx(filename, "{}", out);
}

int TreeliteLoadLightGBMModelEx(char const* filename, char const* config_json, ModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadLightGBMModel(filename);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModel(char const* filename, ModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadXGBoostModel() is deprecated. Please use "
                        << "TreeliteLoadXGBoostModelEx() instead.";
  return TreeliteLoadXGBoostModelEx(filename, "{}", out);
}

int TreeliteLoadXGBoostModelEx(char const* filename, char const* config_json, ModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadXGBoostModel(filename);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostJSON(char const* filename, ModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadXGBoostJSON() is deprecated. Please use "
                        << "TreeliteLoadXGBoostJSONEx() instead.";
  return TreeliteLoadXGBoostJSONEx(filename, "{}", out);
}

int TreeliteLoadXGBoostJSONEx(char const* filename, char const* config_json, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadXGBoostJSONModel(filename, config_json);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostJSONString(char const* json_str, size_t length, ModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadXGBoostJSONString() is deprecated. Please use "
                        << "TreeliteLoadXGBoostJSONStringEx() instead.";
  return TreeliteLoadXGBoostJSONStringEx(json_str, length, "{}", out);
}

int TreeliteLoadXGBoostJSONStringEx(
    char const* json_str, size_t length, char const* config_json, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model
      = frontend::LoadXGBoostJSONModelString(json_str, length, config_json);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelFromMemoryBuffer(void const* buf, size_t len, ModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadXGBoostModelFromMemoryBuffer() is deprecated. Please use "
                        << "TreeliteLoadXGBoostModelFromMemoryBufferEx() instead.";
  return TreeliteLoadXGBoostModelFromMemoryBufferEx(buf, len, "{}", out);
}

int TreeliteLoadXGBoostModelFromMemoryBufferEx(
    void const* buf, size_t len, char const* config_json, ModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadXGBoostModel(buf, len);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadLightGBMModelFromString(char const* model_str, ModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadLightGBMModelFromString() is deprecated. Please use "
                        << "TreeliteLoadLightGBMModelFromStringEx() instead.";
  return TreeliteLoadLightGBMModelFromStringEx(model_str, "{}", out);
}

int TreeliteLoadLightGBMModelFromStringEx(
    char const* model_str, char const* config_json, ModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadLightGBMModelFromString(model_str);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteBuildModelFromJSONString(
    char const* json_str, char const* config_json, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::BuildModelFromJSONString(json_str, config_json);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnRandomForestRegressor(int n_estimators, int n_features,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, double const** value,
    int64_t const** n_node_samples, double const** weighted_n_node_samples, double const** impurity,
    ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnRandomForestRegressor(n_estimators,
      n_features, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnIsolationForest(int n_estimators, int n_features, int64_t const* node_count,
    int64_t const** children_left, int64_t const** children_right, int64_t const** feature,
    double const** threshold, double const** value, int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double const ratio_c,
    ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnIsolationForest(n_estimators, n_features,
      node_count, children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity, ratio_c);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnRandomForestClassifier(int n_estimators, int n_features, int n_classes,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, double const** value,
    int64_t const** n_node_samples, double const** weighted_n_node_samples, double const** impurity,
    ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnRandomForestClassifier(n_estimators,
      n_features, n_classes, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnGradientBoostingRegressor(int n_iter, int n_features,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, double const** value,
    int64_t const** n_node_samples, double const** weighted_n_node_samples, double const** impurity,
    ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnGradientBoostingRegressor(n_iter, n_features,
      node_count, children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnGradientBoostingClassifier(int n_iter, int n_features, int n_classes,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, double const** value,
    int64_t const** n_node_samples, double const** weighted_n_node_samples, double const** impurity,
    ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnGradientBoostingClassifier(n_iter, n_features,
      n_classes, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnHistGradientBoostingRegressor(int n_iter, int n_features,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, int8_t const** default_left,
    double const** value, int64_t const** n_node_samples, double const** gain,
    double const* baseline_prediction, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnHistGradientBoostingRegressor(n_iter,
      n_features, node_count, children_left, children_right, feature, threshold, default_left,
      value, n_node_samples, gain, baseline_prediction);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnHistGradientBoostingClassifier(int n_iter, int n_features, int n_classes,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, int8_t const** default_left,
    double const** value, int64_t const** n_node_samples, double const** gain,
    double const* baseline_prediction, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnHistGradientBoostingClassifier(n_iter,
      n_features, n_classes, node_count, children_left, children_right, feature, threshold,
      default_left, value, n_node_samples, gain, baseline_prediction);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteSerializeModel(char const* filename, ModelHandle handle) {
  API_BEGIN();
  TREELITE_LOG(WARNING) << "TreeliteSerializeModel() is deprecated; "
                        << "please use TreeliteSerializeModelToFile() instead";
  return TreeliteSerializeModelToFile(handle, filename);
  API_END();
}

int TreeliteDeserializeModel(char const* filename, ModelHandle* out) {
  API_BEGIN();
  TREELITE_LOG(WARNING) << "TreeliteDeserializeModel() is deprecated; "
                        << "please use TreeliteDeserializeModelFromFile() instead";
  return TreeliteDeserializeModelFromFile(filename, out);
  API_END();
}

int TreeliteSerializeModelToFile(ModelHandle handle, char const* filename) {
  API_BEGIN();
  std::ofstream ofs(filename, std::ios::out | std::ios::binary);
  TREELITE_CHECK(ofs) << "Failed to open file '" << filename << "'";
  ofs.exceptions(std::ios::failbit | std::ios::badbit);  // throw exception on failure
  auto* model_ = static_cast<Model*>(handle);
  model_->SerializeToStream(ofs);
  API_END();
}

int TreeliteDeserializeModelFromFile(char const* filename, ModelHandle* out) {
  API_BEGIN();
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  TREELITE_CHECK(ifs) << "Failed to open file '" << filename << "'";
  ifs.exceptions(std::ios::failbit | std::ios::badbit);  // throw exception on failure
  std::unique_ptr<Model> model = Model::DeserializeFromStream(ifs);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteSerializeModelToBytes(
    ModelHandle handle, char const** out_bytes, size_t* out_bytes_len) {
  API_BEGIN();
  std::ostringstream oss;
  oss.exceptions(std::ios::failbit | std::ios::badbit);  // throw exception on failure
  auto* model_ = static_cast<Model*>(handle);
  model_->SerializeToStream(oss);

  std::string& ret_str = TreeliteAPIThreadLocalStore::Get()->ret_str;
  ret_str = oss.str();
  *out_bytes = ret_str.data();
  *out_bytes_len = ret_str.length();
  API_END();
}

int TreeliteDeserializeModelFromBytes(char const* bytes, size_t bytes_len, ModelHandle* out) {
  API_BEGIN();
  std::istringstream iss(std::string(bytes, bytes_len));
  iss.exceptions(std::ios::failbit | std::ios::badbit);  // throw exception on failure
  std::unique_ptr<Model> model = Model::DeserializeFromStream(iss);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteConcatenateModelObjects(ModelHandle const* objs, size_t len, ModelHandle* out) {
  API_BEGIN();
  std::vector<Model const*> model_objs(len, nullptr);
  std::transform(objs, objs + len, model_objs.begin(),
      [](const ModelHandle e) { return static_cast<const Model*>(e); });
  auto concatenated_model = ConcatenateModelObjects(model_objs);
  *out = static_cast<ModelHandle>(concatenated_model.release());
  API_END();
}

int TreeliteDumpAsJSON(ModelHandle handle, int pretty_print, char const** out_json_str) {
  API_BEGIN();
  auto* model_ = static_cast<Model*>(handle);
  std::string& ret_str = TreeliteAPIThreadLocalStore::Get()->ret_str;
  ret_str = model_->DumpAsJSON(pretty_print != 0);
  *out_json_str = ret_str.c_str();
  API_END();
}

int TreeliteFreeModel(ModelHandle handle) {
  API_BEGIN();
  delete static_cast<Model*>(handle);
  API_END();
}

int TreeliteGTILParseConfig(char const* config_json, GTILConfigHandle* out) {
  API_BEGIN();
  auto parsed_config = std::make_unique<gtil::Configuration>(config_json);
  *out = static_cast<GTILConfigHandle>(parsed_config.release());
  API_END();
}

int TreeliteGTILDeleteConfig(GTILConfigHandle handle) {
  API_BEGIN();
  delete static_cast<gtil::Configuration*>(handle);
  API_END();
}

int TreeliteGTILGetPredictOutputSize(ModelHandle model, size_t num_row, size_t* out) {
  API_BEGIN();
  TREELITE_LOG(WARNING) << "TreeliteGTILGetPredictOutputSize() is deprecated; "
                        << "please use TreeliteGTILGetPredictOutputSizeEx() instead";
  auto const* model_ = static_cast<Model const*>(model);
  *out = gtil::GetPredictOutputSize(model_, num_row, gtil::Configuration{});
  API_END();
}

int TreeliteGTILGetPredictOutputSizeEx(
    ModelHandle model, size_t num_row, GTILConfigHandle config, size_t* out) {
  API_BEGIN();
  auto const* model_ = static_cast<Model const*>(model);
  auto const* config_ = static_cast<gtil::Configuration const*>(config);
  *out = gtil::GetPredictOutputSize(model_, num_row, *config_);
  API_END();
}

int TreeliteGTILPredict(ModelHandle model, float const* input, size_t num_row, float* output,
    int nthread, int pred_transform, size_t* out_result_size) {
  API_BEGIN();
  TREELITE_LOG(WARNING)
      << "TreeliteGTILPredict() is deprecated; please use TreeliteGTILPredictEx() instead.";
  auto const* model_ = static_cast<Model const*>(model);
  gtil::Configuration config;
  config.pred_type = (pred_transform == 1 ? treelite::gtil::PredictType::kPredictDefault
                                          : treelite::gtil::PredictType::kPredictRaw);
  config.nthread = nthread;
  auto& pred_shape = TreeliteAPIThreadLocalStore::Get()->prediction_shape;
  *out_result_size = gtil::Predict(model_, input, num_row, output, config, pred_shape);
  API_END();
}

int TreeliteGTILPredictEx(ModelHandle model, float const* input, size_t num_row, float* output,
    GTILConfigHandle config, size_t* out_result_size, size_t* out_result_ndim,
    size_t** out_result_shape) {
  API_BEGIN();
  auto const* model_ = static_cast<Model const*>(model);
  auto const* config_ = static_cast<gtil::Configuration const*>(config);
  auto& pred_shape = TreeliteAPIThreadLocalStore::Get()->prediction_shape;
  *out_result_size = gtil::Predict(model_, input, num_row, output, *config_, pred_shape);
  auto prod = std::accumulate<>(
      std::begin(pred_shape), std::end(pred_shape), std::size_t(1), std::multiplies<>{});
  TREELITE_CHECK_EQ(prod, *out_result_size);
  *out_result_ndim = pred_shape.size();
  *out_result_shape = pred_shape.data();
  API_END();
}

int TreeliteQueryNumTree(ModelHandle handle, size_t* out) {
  API_BEGIN();
  auto const* model_ = static_cast<Model const*>(handle);
  *out = model_->GetNumTree();
  API_END();
}

int TreeliteQueryNumFeature(ModelHandle handle, size_t* out) {
  API_BEGIN();
  auto const* model_ = static_cast<Model const*>(handle);
  *out = static_cast<size_t>(model_->num_feature);
  API_END();
}

int TreeliteQueryNumClass(ModelHandle handle, size_t* out) {
  API_BEGIN();
  auto const* model_ = static_cast<Model const*>(handle);
  *out = static_cast<size_t>(model_->task_param.num_class);
  API_END();
}

int TreeliteSetTreeLimit(ModelHandle handle, size_t limit) {
  API_BEGIN();
  TREELITE_CHECK_GT(limit, 0) << "limit should be greater than 0!";
  auto* model_ = static_cast<Model*>(handle);
  const size_t num_tree = model_->GetNumTree();
  TREELITE_CHECK_GE(num_tree, limit) << "Model contains fewer trees(" << num_tree << ") than limit";
  model_->SetTreeLimit(limit);
  API_END();
}

int TreeliteTreeBuilderCreateValue(void const* init_value, char const* type, ValueHandle* out) {
  API_BEGIN();
  std::unique_ptr<frontend::Value> value = std::make_unique<frontend::Value>();
  *value = frontend::Value::Create(init_value, GetTypeInfoByName(type));
  *out = static_cast<ValueHandle>(value.release());
  API_END();
}

int TreeliteTreeBuilderDeleteValue(ValueHandle handle) {
  API_BEGIN();
  delete static_cast<frontend::Value*>(handle);
  API_END();
}

int TreeliteCreateTreeBuilder(
    char const* threshold_type, char const* leaf_output_type, TreeBuilderHandle* out) {
  API_BEGIN();
  std::unique_ptr<frontend::TreeBuilder> builder{new frontend::TreeBuilder(
      GetTypeInfoByName(threshold_type), GetTypeInfoByName(leaf_output_type))};
  *out = static_cast<TreeBuilderHandle>(builder.release());
  API_END();
}

int TreeliteDeleteTreeBuilder(TreeBuilderHandle handle) {
  API_BEGIN();
  delete static_cast<frontend::TreeBuilder*>(handle);
  API_END();
}

int TreeliteTreeBuilderCreateNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->CreateNode(node_key);
  API_END();
}

int TreeliteTreeBuilderDeleteNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->DeleteNode(node_key);
  API_END();
}

int TreeliteTreeBuilderSetRootNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetRootNode(node_key);
  API_END();
}

int TreeliteTreeBuilderSetNumericalTestNode(TreeBuilderHandle handle, int node_key,
    unsigned feature_id, char const* opname, ValueHandle threshold, int default_left,
    int left_child_key, int right_child_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetNumericalTestNode(node_key, feature_id, opname,
      *static_cast<frontend::Value const*>(threshold), (default_left != 0), left_child_key,
      right_child_key);
  API_END();
}

int TreeliteTreeBuilderSetCategoricalTestNode(TreeBuilderHandle handle, int node_key,
    unsigned feature_id, unsigned int const* left_categories, size_t left_categories_len,
    int default_left, int left_child_key, int right_child_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  std::vector<uint32_t> vec(left_categories_len);
  for (size_t i = 0; i < left_categories_len; ++i) {
    TREELITE_CHECK(left_categories[i] <= std::numeric_limits<uint32_t>::max());
    vec[i] = static_cast<uint32_t>(left_categories[i]);
  }
  builder->SetCategoricalTestNode(
      node_key, feature_id, vec, (default_left != 0), left_child_key, right_child_key);
  API_END();
}

int TreeliteTreeBuilderSetLeafNode(TreeBuilderHandle handle, int node_key, ValueHandle leaf_value) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetLeafNode(node_key, *static_cast<frontend::Value const*>(leaf_value));
  API_END();
}

int TreeliteTreeBuilderSetLeafVectorNode(TreeBuilderHandle handle, int node_key,
    ValueHandle const* leaf_vector, size_t leaf_vector_len) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  std::vector<frontend::Value> vec(leaf_vector_len);
  TREELITE_CHECK(leaf_vector) << "leaf_vector argument must not be null";
  for (size_t i = 0; i < leaf_vector_len; ++i) {
    TREELITE_CHECK(leaf_vector[i]) << "leaf_vector[" << i << "] contains an empty Value handle";
    vec[i] = *static_cast<frontend::Value const*>(leaf_vector[i]);
  }
  builder->SetLeafVectorNode(node_key, vec);
  API_END();
}

int TreeliteCreateModelBuilder(int num_feature, int num_class, int average_tree_output,
    char const* threshold_type, char const* leaf_output_type, ModelBuilderHandle* out) {
  API_BEGIN();
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(num_feature, num_class, (average_tree_output != 0),
          GetTypeInfoByName(threshold_type), GetTypeInfoByName(leaf_output_type))};
  *out = static_cast<ModelBuilderHandle>(builder.release());
  API_END();
}

int TreeliteModelBuilderSetModelParam(
    ModelBuilderHandle handle, char const* name, char const* value) {
  API_BEGIN();
  auto* builder = static_cast<frontend::ModelBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  builder->SetModelParam(name, value);
  API_END();
}

int TreeliteDeleteModelBuilder(ModelBuilderHandle handle) {
  API_BEGIN();
  delete static_cast<frontend::ModelBuilder*>(handle);
  API_END();
}

int TreeliteModelBuilderInsertTree(
    ModelBuilderHandle handle, TreeBuilderHandle tree_builder_handle, int index) {
  API_BEGIN();
  auto* model_builder = static_cast<frontend::ModelBuilder*>(handle);
  TREELITE_CHECK(model_builder) << "Detected dangling reference to deleted ModelBuilder object";
  auto* tree_builder = static_cast<frontend::TreeBuilder*>(tree_builder_handle);
  TREELITE_CHECK(tree_builder) << "Detected dangling reference to deleted TreeBuilder object";
  return model_builder->InsertTree(tree_builder, index);
  API_END();
}

int TreeliteModelBuilderGetTree(ModelBuilderHandle handle, int index, TreeBuilderHandle* out) {
  API_BEGIN();
  auto* model_builder = static_cast<frontend::ModelBuilder*>(handle);
  TREELITE_CHECK(model_builder) << "Detected dangling reference to deleted ModelBuilder object";
  auto* tree_builder = model_builder->GetTree(index);
  TREELITE_CHECK(tree_builder) << "Detected dangling reference to deleted TreeBuilder object";
  *out = static_cast<TreeBuilderHandle>(tree_builder);
  API_END();
}

int TreeliteModelBuilderDeleteTree(ModelBuilderHandle handle, int index) {
  API_BEGIN();
  auto* builder = static_cast<frontend::ModelBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  builder->DeleteTree(index);
  API_END();
}

int TreeliteModelBuilderCommitModel(ModelBuilderHandle handle, ModelHandle* out) {
  API_BEGIN();
  auto* builder = static_cast<frontend::ModelBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  std::unique_ptr<Model> model = builder->CommitModel();
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteRegisterLogCallback(void (*callback)(char const*)) {
  API_BEGIN();
  LogCallbackRegistry* registry = LogCallbackRegistryStore::Get();
  registry->RegisterCallBackLogInfo(callback);
  API_END();
}

int TreeliteRegisterWarningCallback(void (*callback)(char const*)) {
  API_BEGIN();
  LogCallbackRegistry* registry = LogCallbackRegistryStore::Get();
  registry->RegisterCallBackLogWarning(callback);
  API_END();
}
