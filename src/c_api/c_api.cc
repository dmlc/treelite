/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file c_api.cc
 * \author Hyunsu Cho
 * \brief C API of treelite, used for interfacing with other languages
 */


#include <treelite/annotator.h>
#include <treelite/c_api.h>
#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <treelite/data.h>
#include <treelite/filesystem.h>
#include <treelite/frontend.h>
#include <treelite/math.h>
#include <dmlc/thread_local.h>
#include <memory>
#include <algorithm>
#include "./c_api_error.h"

using namespace treelite;

namespace {

struct CompilerHandleImpl {
  std::string name;
  std::vector<std::pair<std::string, std::string>> cfg;
  std::unique_ptr<Compiler> compiler;
  explicit CompilerHandleImpl(const std::string& name)
    : name(name), cfg(), compiler(nullptr) {}
  ~CompilerHandleImpl() = default;
};

/*! \brief entry to to easily hold returning information */
struct TreeliteAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
};

// define threadlocal store for returning information
using TreeliteAPIThreadLocalStore
  = dmlc::ThreadLocalStore<TreeliteAPIThreadLocalEntry>;

}  // anonymous namespace

int TreeliteDMatrixCreateFromFile(const char* path,
                                  const char* format,
                                  int nthread,
                                  int verbose,
                                  DMatrixHandle* out) {
  API_BEGIN();
  *out = static_cast<DMatrixHandle>(DMatrix::Create(path, format,
                                    nthread, verbose));
  API_END();
}

int TreeliteDMatrixCreateFromCSR(const float* data,
                                 const unsigned* col_ind,
                                 const size_t* row_ptr,
                                 size_t num_row,
                                 size_t num_col,
                                 DMatrixHandle* out) {
  API_BEGIN();
  std::unique_ptr<DMatrix> dmat{new DMatrix()};
  dmat->Clear();
  auto& data_ = dmat->data;
  auto& col_ind_ = dmat->col_ind;
  auto& row_ptr_ = dmat->row_ptr;
  data_.reserve(row_ptr[num_row]);
  col_ind_.reserve(row_ptr[num_row]);
  row_ptr_.reserve(num_row + 1);
  for (size_t i = 0; i < num_row; ++i) {
    const size_t jbegin = row_ptr[i];
    const size_t jend = row_ptr[i + 1];
    for (size_t j = jbegin; j < jend; ++j) {
      if (!math::CheckNAN(data[j])) {  // skip NaN
        data_.push_back(data[j]);
        CHECK_LT(col_ind[j], std::numeric_limits<uint32_t>::max())
          << "feature index too big to fit into uint32_t";
        col_ind_.push_back(static_cast<uint32_t>(col_ind[j]));
      }
    }
    row_ptr_.push_back(data_.size());
  }
  data_.shrink_to_fit();
  col_ind_.shrink_to_fit();
  dmat->num_row = num_row;
  dmat->num_col = num_col;
  dmat->nelem = data_.size();  // some nonzeros may have been deleted as NAN

  *out = static_cast<DMatrixHandle>(dmat.release());
  API_END();
}

int TreeliteDMatrixCreateFromMat(const float* data,
                                 size_t num_row,
                                 size_t num_col,
                                 float missing_value,
                                 DMatrixHandle* out) {
  const bool nan_missing = math::CheckNAN(missing_value);
  API_BEGIN();
  CHECK_LT(num_col, std::numeric_limits<uint32_t>::max())
    << "num_col argument is too big";
  std::unique_ptr<DMatrix> dmat{new DMatrix()};
  dmat->Clear();
  auto& data_ = dmat->data;
  auto& col_ind_ = dmat->col_ind;
  auto& row_ptr_ = dmat->row_ptr;
  // make an educated guess for initial sizes,
  // so as to present initial wave of allocation
  const size_t guess_size
    = std::min(std::min(num_row * num_col, num_row * 1000),
               static_cast<size_t>(64 * 1024 * 1024));
  data_.reserve(guess_size);
  col_ind_.reserve(guess_size);
  row_ptr_.reserve(num_row + 1);
  const float* row = &data[0];  // points to beginning of each row
  for (size_t i = 0; i < num_row; ++i, row += num_col) {
    for (size_t j = 0; j < num_col; ++j) {
      if (math::CheckNAN(row[j])) {
        CHECK(nan_missing)
          << "The missing_value argument must be set to NaN if there is any "
          << "NaN in the matrix.";
      } else if (nan_missing || row[j] != missing_value) {
        // row[j] is a valid entry
        data_.push_back(row[j]);
        col_ind_.push_back(static_cast<uint32_t>(j));
      }
    }
    row_ptr_.push_back(data_.size());
  }
  data_.shrink_to_fit();
  col_ind_.shrink_to_fit();
  dmat->num_row = num_row;
  dmat->num_col = num_col;
  dmat->nelem = data_.size();  // some nonzeros may have been deleted as NaN

  *out = static_cast<DMatrixHandle>(dmat.release());
  API_END();
}

int TreeliteDMatrixGetDimension(DMatrixHandle handle,
                                size_t* out_num_row,
                                size_t* out_num_col,
                                size_t* out_nelem) {
  API_BEGIN();
  const DMatrix* dmat = static_cast<DMatrix*>(handle);
  *out_num_row = dmat->num_row;
  *out_num_col = dmat->num_col;
  *out_nelem = dmat->nelem;
  API_END();
}

int TreeliteDMatrixGetPreview(DMatrixHandle handle,
                              const char** out_preview) {
  API_BEGIN();
  const DMatrix* dmat = static_cast<DMatrix*>(handle);
  std::string& ret_str = TreeliteAPIThreadLocalStore::Get()->ret_str;
  std::ostringstream oss;
  const size_t iend = (dmat->nelem <= 50) ? dmat->nelem : 25;
  for (size_t i = 0; i < iend; ++i) {
    const size_t row_ind =
      std::upper_bound(&dmat->row_ptr[0], &dmat->row_ptr[dmat->num_row + 1], i)
        - &dmat->row_ptr[0] - 1;
    oss << "  (" << row_ind << ", " << dmat->col_ind[i] << ")\t"
        << dmat->data[i] << "\n";
  }
  if (dmat->nelem > 50) {
    oss << "  :\t:\n";
    for (size_t i = dmat->nelem - 25; i < dmat->nelem; ++i) {
      const size_t row_ind =
        std::upper_bound(&dmat->row_ptr[0], &dmat->row_ptr[dmat->num_row + 1], i)
        - &dmat->row_ptr[0] - 1;
      oss << "  (" << row_ind << ", " << dmat->col_ind[i] << ")\t"
        << dmat->data[i] << "\n";
    }
  }
  ret_str = oss.str();
  *out_preview = ret_str.c_str();
  API_END();
}

int TreeliteDMatrixGetArrays(DMatrixHandle handle,
                             const float** out_data,
                             const uint32_t** out_col_ind,
                             const size_t** out_row_ptr) {
  API_BEGIN();
  const DMatrix* dmat_ = static_cast<DMatrix*>(handle);
  *out_data = &dmat_->data[0];
  *out_col_ind = &dmat_->col_ind[0];
  *out_row_ptr = &dmat_->row_ptr[0];
  API_END();
}

int TreeliteDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  delete static_cast<DMatrix*>(handle);
  API_END();
}

int TreeliteAnnotateBranch(ModelHandle model,
                           DMatrixHandle dmat,
                           int nthread,
                           int verbose,
                           AnnotationHandle* out) {
  API_BEGIN();
  std::unique_ptr<BranchAnnotator> annotator{new BranchAnnotator()};
  const Model* model_ = static_cast<Model*>(model);
  const DMatrix* dmat_ = static_cast<DMatrix*>(dmat);
  annotator->Annotate(*model_, dmat_, nthread, verbose);
  *out = static_cast<AnnotationHandle>(annotator.release());
  API_END();
}

int TreeliteAnnotationSave(AnnotationHandle handle,
                           const char* path) {
  API_BEGIN();
  const BranchAnnotator* annotator = static_cast<BranchAnnotator*>(handle);
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(path, "w"));
  annotator->Save(fo.get());
  API_END();
}

int TreeliteAnnotationFree(AnnotationHandle handle) {
  API_BEGIN();
  delete static_cast<BranchAnnotator*>(handle);
  API_END();
}

int TreeliteCompilerCreate(const char* name,
                           CompilerHandle* out) {
  API_BEGIN();
  std::unique_ptr<CompilerHandleImpl> compiler{new CompilerHandleImpl(name)};
  *out = static_cast<CompilerHandle>(compiler.release());
  API_END();
}

int TreeliteCompilerSetParam(CompilerHandle handle,
                             const char* name,
                             const char* value) {
  API_BEGIN();
  CompilerHandleImpl* impl = static_cast<CompilerHandleImpl*>(handle);
  auto& cfg_ = impl->cfg;
  std::string name_(name);
  std::string value_(value);
  // check for duplicate parameters
  auto it = std::find_if(cfg_.begin(), cfg_.end(),
    [&name_](const std::pair<std::string, std::string>& x) {
      return x.first == name_;
    });
  if (it == cfg_.end()) {
    cfg_.emplace_back(name_, value_);
  } else {
    it->second = value;
  }
  API_END();
}

int TreeliteCompilerGenerateCode(CompilerHandle compiler,
                                 ModelHandle model,
                                 int verbose,
                                 const char* dirpath) {
  API_BEGIN();
  if (verbose > 0) {  // verbose enabled
    int ret = TreeliteCompilerSetParam(compiler, "verbose",
                                       std::to_string(verbose).c_str());
    if (ret < 0) {  // SetParam failed
      return ret;
    }
  }
  const Model* model_ = static_cast<Model*>(model);
  CompilerHandleImpl* impl = static_cast<CompilerHandleImpl*>(compiler);

  // create directory named dirpath
  const std::string& dirpath_(dirpath);
  filesystem::CreateDirectoryIfNotExist(dirpath);

  compiler::CompilerParam cparam;
  cparam.Init(impl->cfg, dmlc::parameter::kAllMatch);

  /* compile model */
  impl->compiler.reset(Compiler::Create(impl->name, cparam));
  auto compiled_model = impl->compiler->Compile(*model_);
  if (verbose > 0) {
    LOG(INFO) << "Code generation finished. Writing code to files...";
  }

  for (const auto& it : compiled_model.files) {
    if (verbose > 0) {
      LOG(INFO) << "Writing file " << it.first << "...";
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
  delete static_cast<CompilerHandleImpl*>(handle);
  API_END();
}

int TreeliteLoadLightGBMModel(const char* filename,
                              ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model{new Model()};
  frontend::LoadLightGBMModel(filename, model.get());
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModel(const char* filename,
                             ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model{new Model()};
  frontend::LoadXGBoostModel(filename, model.get());
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelFromMemoryBuffer(const void* buf, size_t len,
                                             ModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<Model> model{new Model()};
  frontend::LoadXGBoostModel(buf, len, model.get());
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}

int TreeliteFreeModel(ModelHandle handle) {
  API_BEGIN();
  delete static_cast<Model*>(handle);
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
  *out = static_cast<size_t>(model_->GetNumFeature());
  API_END();
}

int TreeliteQueryNumOutputGroups(ModelHandle handle, size_t* out) {
  API_BEGIN();
  const auto* model_ = static_cast<const Model*>(handle);
  *out = static_cast<size_t>(model_->GetNumOutputGroup());
  API_END();
}

int TreeliteSetTreeLimit(ModelHandle handle, size_t limit) {
  API_BEGIN();
  CHECK_GT(limit, 0) << "limit should be greater than 0!";
  auto* model_ = static_cast<Model*>(handle);
  const size_t num_tree = model_->GetNumTree();
  CHECK_GE(num_tree, limit) << "Model contains less trees(" << num_tree << ") than limit";
  model_->SetTreeLimit(limit);
  API_END();
}

int TreeliteTreeBuilderCreateValue(const void* init_value, const char* type, ValueHandle* out) {
  API_BEGIN();
  std::unique_ptr<frontend::Value> value = std::make_unique<frontend::Value>();
  *value = frontend::Value::Create(init_value, typeinfo_table.at(type));
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
    new frontend::TreeBuilder(typeinfo_table.at(threshold_type),
                              typeinfo_table.at(leaf_output_type))
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
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->CreateNode(node_key);
  API_END();
}

int TreeliteTreeBuilderDeleteNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->DeleteNode(node_key);
  API_END();
}

int TreeliteTreeBuilderSetRootNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetRootNode(node_key);
  API_END();
}

int TreeliteTreeBuilderSetNumericalTestNode(
    TreeBuilderHandle handle, int node_key, unsigned feature_id, const char* opname,
    ValueHandle threshold, int default_left, int left_child_key, int right_child_key) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
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
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  std::vector<uint32_t> vec(left_categories_len);
  for (size_t i = 0; i < left_categories_len; ++i) {
    CHECK(left_categories[i] <= std::numeric_limits<uint32_t>::max());
    vec[i] = static_cast<uint32_t>(left_categories[i]);
  }
  builder->SetCategoricalTestNode(node_key, feature_id, vec, (default_left != 0),
                                  left_child_key, right_child_key);
  API_END();
}

int TreeliteTreeBuilderSetLeafNode(TreeBuilderHandle handle, int node_key, ValueHandle leaf_value) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetLeafNode(node_key, *static_cast<const frontend::Value*>(leaf_value));
  API_END();
}

int TreeliteTreeBuilderSetLeafVectorNode(TreeBuilderHandle handle, int node_key,
                                         const ValueHandle* leaf_vector, size_t leaf_vector_len) {
  API_BEGIN();
  auto* builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  std::vector<frontend::Value> vec(leaf_vector_len);
  for (size_t i = 0; i < leaf_vector_len; ++i) {
    vec[i] = *static_cast<const frontend::Value*>(leaf_vector[i]);
  }
  builder->SetLeafVectorNode(node_key, vec);
  API_END();
}

int TreeliteCreateModelBuilder(
    int num_feature, int num_output_group, int random_forest_flag, const char* threshold_type,
    const char* leaf_output_type, ModelBuilderHandle* out) {
  API_BEGIN();
  std::unique_ptr<frontend::ModelBuilder> builder{new frontend::ModelBuilder(
      num_feature, num_output_group, (random_forest_flag != 0), typeinfo_table.at(threshold_type),
      typeinfo_table.at(leaf_output_type))};
  *out = static_cast<ModelBuilderHandle>(builder.release());
  API_END();
}

int TreeliteModelBuilderSetModelParam(ModelBuilderHandle handle,
                                      const char* name,
                                      const char* value) {
  API_BEGIN();
  auto* builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  builder->SetModelParam(name, value);
  API_END();
}

int TreeliteDeleteModelBuilder(ModelBuilderHandle handle) {
  API_BEGIN();
  delete static_cast<frontend::ModelBuilder*>(handle);
  API_END();
}

int TreeliteModelBuilderInsertTree(ModelBuilderHandle handle,
                                   TreeBuilderHandle tree_builder_handle,
                                   int index) {
  API_BEGIN();
  auto* model_builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(model_builder) << "Detected dangling reference to deleted ModelBuilder object";
  auto* tree_builder = static_cast<frontend::TreeBuilder*>(tree_builder_handle);
  CHECK(tree_builder) << "Detected dangling reference to deleted TreeBuilder object";
  return model_builder->InsertTree(tree_builder, index);
  API_END();
}

int TreeliteModelBuilderGetTree(ModelBuilderHandle handle, int index,
                                TreeBuilderHandle *out) {
  API_BEGIN();
  auto* model_builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(model_builder) << "Detected dangling reference to deleted ModelBuilder object";
  auto* tree_builder = model_builder->GetTree(index);
  CHECK(tree_builder) << "Detected dangling reference to deleted TreeBuilder object";
  *out = static_cast<TreeBuilderHandle>(tree_builder);
  API_END();
}

int TreeliteModelBuilderDeleteTree(ModelBuilderHandle handle, int index) {
  API_BEGIN();
  auto* builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  builder->DeleteTree(index);
  API_END();
}

int TreeliteModelBuilderCommitModel(ModelBuilderHandle handle,
                                    ModelHandle* out) {
  API_BEGIN();
  auto* builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  std::unique_ptr<Model> model{new Model()};
  builder->CommitModel(model.get());
  *out = static_cast<ModelHandle>(model.release());
  API_END();
}
