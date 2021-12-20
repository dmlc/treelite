/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file c_api.cc
 * \author Hyunsu Cho
 * \brief C API of treelite, used for interfacing with other languages
 */

#include <treelite/annotator.h>
#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <treelite/data.h>
#include <treelite/filesystem.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <treelite/math.h>
#include <treelite/gtil.h>
#include <treelite/logging.h>
#include <memory>
#include <algorithm>
#include <fstream>
#include <string>
#include <cstdio>

using namespace treelite;

namespace {

/*! \brief entry to to easily hold returning information */
struct TreeliteAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
};

// define threadlocal store for returning information
using TreeliteAPIThreadLocalStore = ThreadLocalStore<TreeliteAPIThreadLocalEntry>;

}  // anonymous namespace

int TreeliteAnnotateBranch(
    ModelHandle model, DMatrixHandle dmat, int nthread, int verbose, AnnotationHandle* out) {
  API_BEGIN();
  std::unique_ptr<BranchAnnotator> annotator{new BranchAnnotator()};
  const Model* model_ = static_cast<Model*>(model);
  const auto* dmat_ = static_cast<const DMatrix*>(dmat);
  TREELITE_CHECK(dmat_) << "Found a dangling reference to DMatrix";
  annotator->Annotate(*model_, dmat_, nthread, verbose);
  *out = static_cast<AnnotationHandle>(annotator.release());
  API_END();
}

int TreeliteAnnotationSave(AnnotationHandle handle,
                           const char* path) {
  API_BEGIN();
  const BranchAnnotator* annotator = static_cast<BranchAnnotator*>(handle);
  std::ofstream fo(path);
  annotator->Save(fo);
  API_END();
}

int TreeliteAnnotationFree(AnnotationHandle handle) {
  API_BEGIN();
  delete static_cast<BranchAnnotator*>(handle);
  API_END();
}

int TreeliteCompilerCreateV2(const char* name, const char* params_json_str, CompilerHandle* out) {
  API_BEGIN();
  std::unique_ptr<Compiler> compiler{Compiler::Create(name, params_json_str)};
  *out = static_cast<CompilerHandle>(compiler.release());
  API_END();
}

int TreeliteCompilerGenerateCodeV2(CompilerHandle compiler,
                                   ModelHandle model,
                                   const char* dirpath) {
  API_BEGIN();
  const Model* model_ = static_cast<Model*>(model);
  Compiler* compiler_ = static_cast<Compiler*>(compiler);
  TREELITE_CHECK(model_);
  TREELITE_CHECK(compiler_);
  compiler::CompilerParam param = compiler_->QueryParam();

  // create directory named dirpath
  const std::string& dirpath_(dirpath);
  filesystem::CreateDirectoryIfNotExist(dirpath);

  /* compile model */
  auto compiled_model = compiler_->Compile(*model_);
  if (param.verbose > 0) {
    TREELITE_LOG(INFO) << "Code generation finished. Writing code to files...";
  }

  for (const auto& it : compiled_model.files) {
    if (param.verbose > 0) {
      TREELITE_LOG(INFO) << "Writing file " << it.first << "...";
    }
    const std::string filename_full = dirpath_ + "/" + it.first;
    if (it.second.is_binary) {
      filesystem::WriteToFile(filename_full, it.second.content_binary);
    } else {
      filesystem::WriteToFile(filename_full, it.second.content);
    }
  }

  API_END();
}

int TreeliteCompilerFree(CompilerHandle handle) {
  API_BEGIN();
  delete static_cast<Compiler*>(handle);
  API_END();
}

int TreeliteLoadLightGBMModel(const char* filename, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadLightGBMModel(filename);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModel(const char* filename, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadXGBoostModel(filename);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostJSON(const char* filename, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadXGBoostJSONModel(filename);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostJSONString(const char* json_str, size_t length, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadXGBoostJSONModelString(json_str, length);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelFromMemoryBuffer(const void* buf, size_t len, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadXGBoostModel(buf, len);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnRandomForestRegressor(
    int n_estimators, int n_features, const int64_t* node_count, const int64_t** children_left,
    const int64_t** children_right, const int64_t** feature, const double** threshold,
    const double** value, const int64_t** n_node_samples, const double** weighted_n_node_samples,
    const double** impurity, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnRandomForestRegressor(
      n_estimators, n_features, node_count, children_left, children_right, feature, threshold,
      value, n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnIsolationForest(
    int n_estimators, int n_features, const int64_t* node_count, const int64_t** children_left,
    const int64_t** children_right, const int64_t** feature, const double** threshold,
    const double** value, const int64_t** n_node_samples, const double** weighted_n_node_samples,
    const double** impurity, const double ratio_c, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnIsolationForest(
      n_estimators, n_features, node_count, children_left, children_right, feature, threshold,
      value, n_node_samples, weighted_n_node_samples, impurity, ratio_c);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnRandomForestClassifier(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnRandomForestClassifier(
      n_estimators, n_features, n_classes, node_count, children_left, children_right, feature,
      threshold, value, n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnGradientBoostingRegressor(
    int n_estimators, int n_features, const int64_t* node_count, const int64_t** children_left,
    const int64_t** children_right, const int64_t** feature, const double** threshold,
    const double** value, const int64_t** n_node_samples, const double** weighted_n_node_samples,
    const double** impurity, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnGradientBoostingRegressor(
      n_estimators, n_features, node_count, children_left, children_right, feature, threshold,
      value, n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnGradientBoostingClassifier(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity, ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model = frontend::LoadSKLearnGradientBoostingClassifier(
      n_estimators, n_features, n_classes, node_count, children_left, children_right, feature,
      threshold, value, n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteSerializeModel(const char* filename, ModelHandle handle) {
  API_BEGIN();
  FILE* fp = std::fopen(filename, "wb");
  TREELITE_CHECK(fp) << "Failed to open file '" << filename << "'";
  auto* model_ = static_cast<Model*>(handle);
  model_->SerializeToFile(fp);
  std::fclose(fp);
  API_END();
}

int TreeliteDeserializeModel(const char* filename, ModelHandle* out) {
  API_BEGIN();
  FILE* fp = std::fopen(filename, "rb");
  TREELITE_CHECK(fp) << "Failed to open file '" << filename << "'";
  std::unique_ptr<Model> model = Model::DeserializeFromFile(fp);
  std::fclose(fp);
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteDumpAsJSON(ModelHandle handle, int pretty_print, const char** out_json_str) {
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

int TreeliteGTILGetPredictOutputSize(ModelHandle handle, size_t num_row, size_t* out) {
  API_BEGIN();
  const auto* model_ = static_cast<const Model*>(handle);
  *out = gtil::GetPredictOutputSize(model_, num_row);
  API_END();
}

int TreeliteGTILPredict(ModelHandle handle, const float* input, size_t num_row, float* output,
                        int pred_transform, size_t* out_result_size) {
  API_BEGIN();
  const auto* model_ = static_cast<const Model*>(handle);
  *out_result_size =
      gtil::Predict(model_, input, num_row, output, (pred_transform == 1));
  API_END();
}

int TreeliteQueryNumTree(ModelHandle handle, size_t* out) {
  API_BEGIN();
  const auto* model_ = static_cast<const Model*>(handle);
  *out = model_->GetNumTree();
  API_END();
}

int TreeliteQueryNumFeature(ModelHandle handle, size_t* out) {
  API_BEGIN();
  const auto* model_ = static_cast<const Model*>(handle);
  *out = static_cast<size_t>(model_->num_feature);
  API_END();
}

int TreeliteQueryNumClass(ModelHandle handle, size_t* out) {
  API_BEGIN();
  const auto* model_ = static_cast<const Model*>(handle);
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

int TreeliteTreeBuilderCreateValue(const void* init_value, const char* type, ValueHandle* out) {
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

int TreeliteCreateTreeBuilder(const char* threshold_type, const char* leaf_output_type,
                              TreeBuilderHandle* out) {
  API_BEGIN();
  std::unique_ptr<frontend::TreeBuilder> builder{
    new frontend::TreeBuilder(GetTypeInfoByName(threshold_type),
                              GetTypeInfoByName(leaf_output_type))
  };
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

int TreeliteTreeBuilderSetNumericalTestNode(
    TreeBuilderHandle handle, int node_key, unsigned feature_id, const char* opname,
    ValueHandle threshold, int default_left, int left_child_key, int right_child_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetNumericalTestNode(node_key, feature_id, opname,
                                *static_cast<const frontend::Value*>(threshold),
                                (default_left != 0), left_child_key, right_child_key);
  API_END();
}

int TreeliteTreeBuilderSetCategoricalTestNode(
    TreeBuilderHandle handle, int node_key, unsigned feature_id,
    const unsigned int* left_categories, size_t left_categories_len, int default_left,
    int left_child_key, int right_child_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  std::vector<uint32_t> vec(left_categories_len);
  for (size_t i = 0; i < left_categories_len; ++i) {
    TREELITE_CHECK(left_categories[i] <= std::numeric_limits<uint32_t>::max());
    vec[i] = static_cast<uint32_t>(left_categories[i]);
  }
  builder->SetCategoricalTestNode(node_key, feature_id, vec, (default_left != 0),
                                  left_child_key, right_child_key);
  API_END();
}

int TreeliteTreeBuilderSetLeafNode(TreeBuilderHandle handle, int node_key, ValueHandle leaf_value) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetLeafNode(node_key, *static_cast<const frontend::Value*>(leaf_value));
  API_END();
}

int TreeliteTreeBuilderSetLeafVectorNode(TreeBuilderHandle handle, int node_key,
                                         const ValueHandle* leaf_vector, size_t leaf_vector_len) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  TREELITE_CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  std::vector<frontend::Value> vec(leaf_vector_len);
  TREELITE_CHECK(leaf_vector) << "leaf_vector argument must not be null";
  for (size_t i = 0; i < leaf_vector_len; ++i) {
    TREELITE_CHECK(leaf_vector[i]) << "leaf_vector[" << i << "] contains an empty Value handle";
    vec[i] = *static_cast<const frontend::Value*>(leaf_vector[i]);
  }
  builder->SetLeafVectorNode(node_key, vec);
  API_END();
}

int TreeliteCreateModelBuilder(
    int num_feature, int num_class, int average_tree_output, const char* threshold_type,
    const char* leaf_output_type, ModelBuilderHandle* out) {
  API_BEGIN();
  std::unique_ptr<frontend::ModelBuilder> builder{new frontend::ModelBuilder(
      num_feature, num_class, (average_tree_output != 0), GetTypeInfoByName(threshold_type),
      GetTypeInfoByName(leaf_output_type))};
  *out = static_cast<ModelBuilderHandle>(builder.release());
  API_END();
}

int TreeliteModelBuilderSetModelParam(ModelBuilderHandle handle, const char* name,
                                      const char* value) {
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

int TreeliteModelBuilderInsertTree(ModelBuilderHandle handle, TreeBuilderHandle tree_builder_handle,
                                   int index) {
  API_BEGIN();
  auto* model_builder = static_cast<frontend::ModelBuilder*>(handle);
  TREELITE_CHECK(model_builder) << "Detected dangling reference to deleted ModelBuilder object";
  auto* tree_builder = static_cast<frontend::TreeBuilder*>(tree_builder_handle);
  TREELITE_CHECK(tree_builder) << "Detected dangling reference to deleted TreeBuilder object";
  return model_builder->InsertTree(tree_builder, index);
  API_END();
}

int TreeliteModelBuilderGetTree(ModelBuilderHandle handle, int index, TreeBuilderHandle *out) {
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
