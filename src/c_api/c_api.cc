/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api.cc
 * \author Philip Cho
 * \brief C API of tree-lite, used for interfacing with other languages
 */

#include <treelite/annotator.h>
#include <treelite/c_api.h>
#include <treelite/compiler.h>
#include <treelite/data.h>
#include <treelite/frontend.h>
#include <treelite/predictor.h>
#include <treelite/semantic.h>
#include <dmlc/thread_local.h>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include "./c_api_error.h"
#include "../compiler/param.h"

using namespace treelite;

/*! \brief entry to to easily hold returning information */
struct TreeliteAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
};

// define threadlocal store for returning information
using TreeliteAPIThreadLocalStore
  = dmlc::ThreadLocalStore<TreeliteAPIThreadLocalEntry>;

namespace {

struct CompilerHandleImpl {
  std::string name;
  std::vector<std::pair<std::string, std::string>> cfg;
  std::unique_ptr<Compiler> compiler;
  CompilerHandleImpl(const std::string& name)
    : name(name), cfg(), compiler(nullptr) {}
  ~CompilerHandleImpl() = default;
};

}  // namespace anonymous

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
  DMatrix* dmat = new DMatrix();
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
      if (!common::CheckNAN(data[j])) {  // skip NaN
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

  *out = static_cast<DMatrixHandle>(dmat);
  API_END();
}

int TreeliteDMatrixCreateFromMat(const float* data,
                                 size_t num_row,
                                 size_t num_col,
                                 float missing_value,
                                 DMatrixHandle* out) {
  const bool nan_missing = common::CheckNAN(missing_value);
  API_BEGIN();
  CHECK_LT(num_col, std::numeric_limits<uint32_t>::max())
    << "num_col argument is too big";
  DMatrix* dmat = new DMatrix();
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
      if (common::CheckNAN(row[j])) {
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

  *out = static_cast<DMatrixHandle>(dmat);
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
  for (size_t i = 0; i < 25; ++i) {
    const size_t row_ind =
      std::upper_bound(&dmat->row_ptr[0], &dmat->row_ptr[dmat->num_row + 1], i)
        - &dmat->row_ptr[0] - 1;
    oss << "  (" << row_ind << ", " << dmat->col_ind[i] << ")\t"
        << dmat->data[i] << "\n";
  }
  oss << "  :\t:\n";
  for (size_t i = dmat->nelem - 25; i < dmat->nelem; ++i) {
    const size_t row_ind =
      std::upper_bound(&dmat->row_ptr[0], &dmat->row_ptr[dmat->num_row + 1], i)
      - &dmat->row_ptr[0] - 1;
    oss << "  (" << row_ind << ", " << dmat->col_ind[i] << ")\t"
      << dmat->data[i] << "\n";
  }
  ret_str = oss.str();
  *out_preview = ret_str.c_str();
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
  BranchAnnotator* annotator = new BranchAnnotator();
  const Model* model_ = static_cast<Model*>(model);
  const DMatrix* dmat_ = static_cast<DMatrix*>(dmat);
  annotator->Annotate(*model_, dmat_, nthread, verbose);
  *out = static_cast<AnnotationHandle>(annotator);
  API_END();
}

int TreeliteAnnotationLoad(const char* path,
                           AnnotationHandle* out) {
  API_BEGIN();
  BranchAnnotator* annotator = new BranchAnnotator();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(path, "r"));
  annotator->Load(fi.get());
  *out = static_cast<AnnotationHandle>(annotator);
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
  *out = static_cast<CompilerHandle>(new CompilerHandleImpl(name));
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
                                 const char* path_prefix) {
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
  std::string path_prefix_(path_prefix);
  compiler::CompilerParam cparam;
  cparam.Init(impl->cfg, dmlc::parameter::kAllMatch);

  /* generate semantic model */
  impl->compiler.reset(Compiler::Create(impl->name, cparam));
  auto semantic_model = impl->compiler->Compile(*model_);
  if (verbose > 0) {
    LOG(INFO) << "Code generation finished. Writing code to files...";
  }

  /* write header */
  const std::string header_filename = path_prefix_ + ".h";
  if (verbose > 0) {
    LOG(INFO) << "Writing " << header_filename << " ...";
  }
  {
    std::vector<std::string> lines;
    common::TransformPushBack(&lines, semantic_model.common_header->Compile(),
      [] (std::string line) {
        return line;
      });
    lines.emplace_back();
    std::ostringstream oss;
    using FunctionEntry = semantic::SemanticModel::FunctionEntry;
    std::copy(semantic_model.function_registry.begin(),
              semantic_model.function_registry.end(),
              std::ostream_iterator<FunctionEntry>(oss));
    lines.push_back(oss.str());
    common::WriteToFile(header_filename, lines);
  }
  /* write source file(s) */
  std::vector<std::string> source_list;
  std::vector<std::string> object_list;
  if (semantic_model.units.size() == 1) {   // single file (translation unit)
    const std::string filename = path_prefix_ + ".c";
    const std::string objname = path_prefix_ + ".o";
    source_list.push_back(common::GetBasename(filename));
    object_list.push_back(common::GetBasename(objname));
    if (verbose > 0) {
      LOG(INFO) << "Writing " << filename << " ...";
    }
    auto lines = semantic_model.units[0].Compile(header_filename);
    common::WriteToFile(filename, lines);
  } else {  // multiple files (translation units)
    for (size_t i = 0; i < semantic_model.units.size(); ++i) {
      const std::string filename = path_prefix_ + std::to_string(i) + ".c";
      const std::string objname = path_prefix_ + std::to_string(i) + ".o";
      source_list.push_back(common::GetBasename(filename));
      object_list.push_back(common::GetBasename(objname));
      if (verbose > 0) {
        LOG(INFO) << "Writing " << filename << " ...";
      }
      auto lines = semantic_model.units[i].Compile(header_filename);
      common::WriteToFile(filename, lines);
    }
  }
  /* write Makefile if on Linux */
#ifdef __linux__
  {
    std::string library_name = common::GetBasename(path_prefix_ + ".so");
    std::ostringstream oss;
    oss << "all: " << library_name << std::endl << std::endl
        << library_name << ": ";
    for (const auto& e : object_list) {
      oss << e << " ";
    }
    oss << std::endl
        << "\tgcc -shared -O3 -o $@ $? -fPIC -std=c99 -flto"
        << std::endl << std::endl;
    for (size_t i = 0; i < object_list.size(); ++i) {
      oss << object_list[i] << ": " << source_list[i] << std::endl
          << "\tgcc -c -O3 -o $@ $? -fPIC -std=c99 -flto" << std::endl;
    }
    oss << std::endl
        << "clean:" << std::endl
        << "\trm -fv " << library_name << " ";
    for (const auto& e : object_list) {
      oss << e << " ";
    }
    common::WriteToFile(path_prefix_ + ".Makefile", {oss.str()});
    if (verbose > 0) {
      LOG(INFO) << "Writing " << path_prefix_ << ".Makefile ...";
    }
  }
#endif
  API_END();
}

int TreeliteCompilerFree(CompilerHandle handle) {
  API_BEGIN();
  delete static_cast<CompilerHandleImpl*>(handle);
  API_END();
}

int TreelitePredictorLoad(const char* library_path,
                          PredictorHandle* out) {
  API_BEGIN();
  Predictor* predictor = new Predictor();
  predictor->Load(library_path);
  *out = static_cast<PredictorHandle>(predictor);
  API_END();
}

int TreelitePredictorPredict(PredictorHandle handle,
                             DMatrixHandle dmat,
                             int nthread,
                             int verbose,
                             float* out_result) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  const DMatrix* dmat_ = static_cast<DMatrix*>(dmat);
  predictor_->Predict(dmat_, nthread, verbose, out_result);
  API_END();
}

int TreelitePredictorQueryResultSize(PredictorHandle handle,
                                     DMatrixHandle dmat,
                                     size_t* out) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  const DMatrix* dmat_ = static_cast<DMatrix*>(dmat);
  *out = predictor_->QueryResultSize(dmat_);
  API_END();
}

int TreelitePredictorQueryNumOutputGroup(PredictorHandle handle, size_t* out) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  *out = predictor_->QueryNumOutputGroup();
  API_END();
}

int TreelitePredictorFree(PredictorHandle handle) {
  API_BEGIN();
  delete static_cast<Predictor*>(handle);
  API_END();
}

int TreeliteLoadLightGBMModel(const char* filename,
                              ModelHandle* out) {
  API_BEGIN();
  Model* model = new Model(std::move(frontend::LoadLightGBMModel(filename)));
  *out = static_cast<ModelHandle>(model);
  API_END();
}

int TreeliteLoadXGBoostModel(const char* filename,
                             ModelHandle* out) {
  API_BEGIN();
  Model* model = new Model(std::move(frontend::LoadXGBoostModel(filename)));
  *out = static_cast<ModelHandle>(model);
  API_END();
}

int TreeliteLoadProtobufModel(const char* filename,
                              ModelHandle* out) {
  API_BEGIN();
  Model* model = new Model(std::move(frontend::LoadProtobufModel(filename)));
  *out = static_cast<ModelHandle>(model);
  API_END();
}

int TreeliteCreateModelBuilder(int num_features,
                               ModelBuilderHandle* out) {
  API_BEGIN();
  auto builder = new frontend::ModelBuilder(num_features);
  *out = static_cast<ModelBuilderHandle>(builder);
  API_END();
}

int TreeliteSetModelParam(ModelBuilderHandle handle,
                          const char* name,
                          const char* value) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  builder->SetModelParam(name, value);
  API_END();
}

int TreeliteDeleteModelBuilder(ModelBuilderHandle handle) {
  API_BEGIN();
  delete static_cast<frontend::ModelBuilder*>(handle);
  API_END();
}

int TreeliteCreateTree(ModelBuilderHandle handle, int index) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  return builder->CreateTree(index);
  API_END();
}

int TreeliteDeleteTree(ModelBuilderHandle handle, int index) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  return (builder->DeleteTree(index)) ? 0 : -1;
  API_END();
}

int TreeliteCreateNode(ModelBuilderHandle handle,
                       int tree_index, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  return (builder->CreateNode(tree_index, node_key)) ? 0 : -1;
  API_END();
}

int TreeliteDeleteNode(ModelBuilderHandle handle,
                       int tree_index, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  return (builder->DeleteNode(tree_index, node_key)) ? 0 : -1;
  API_END();
}

int TreeliteSetRootNode(ModelBuilderHandle handle,
                        int tree_index, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  return (builder->SetRootNode(tree_index, node_key)) ? 0 : -1;
  API_END();
}

int TreeliteSetNumericalTestNode(ModelBuilderHandle handle,
                                 int tree_index, int node_key,
                                 unsigned feature_id, const char* opname,
                                 float threshold, int default_left,
                                 int left_child_key, int right_child_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK_GT(optable.count(opname), 0)
    << "No operator `" << opname << "\" exists";
  return (builder->SetNumericalTestNode(tree_index, node_key, feature_id,
                                        optable.at(opname),
                                        static_cast<tl_float>(threshold),
                                        static_cast<bool>(default_left),
                                        left_child_key, right_child_key)) \
                                        ? 0 : -1;
  API_END();
}

int TreeliteSetCategoricalTestNode(ModelBuilderHandle handle,
                                   int tree_index, int node_key,
                                   unsigned feature_id,
                                   const unsigned char* left_categories,
                                   size_t left_categories_len,
                                   int default_left,
                                   int left_child_key, int right_child_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  std::vector<uint8_t> vec(left_categories_len);
  for (size_t i = 0; i < left_categories_len; ++i) {
    CHECK(left_categories[i] <= std::numeric_limits<uint8_t>::max());
    vec[i] = static_cast<uint8_t>(left_categories[i]);
  }
  return (builder->SetCategoricalTestNode(tree_index, node_key, feature_id, vec,
                                          static_cast<bool>(default_left),
                                          left_child_key, right_child_key)) \
                                        ? 0 : -1;
  API_END();
}

int TreeliteSetLeafNode(ModelBuilderHandle handle,
                        int tree_index, int node_key,
                        float leaf_value) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  return (builder->SetLeafNode(tree_index, node_key,
                               static_cast<tl_float>(leaf_value))) ? 0 : -1;
  API_END();
}

int TreeliteCommitModel(ModelBuilderHandle handle,
                        ModelHandle* out) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  Model* model = new Model();
  return (builder->CommitModel(model)) ? 0 : -1;
  API_END();
}
