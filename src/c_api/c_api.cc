/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api.cc
 * \author Philip Cho
 * \brief C API of tree-lite, used for interfacing with other languages
 */

#include <treelite/annotator.h>
#include <treelite/c_api.h>
#include <treelite/compiler.h>
#include <treelite/frontend.h>
#include <treelite/semantic.h>
#include <dmlc/json.h>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include "./c_api_error.h"
#include "../compiler/param.h"
#include "../common/filesystem.h"

using namespace treelite;

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
  common::filesystem::CreateDirectoryIfNotExist(dirpath);
  const std::string basename = common::filesystem::GetBasename(dirpath);

  compiler::CompilerParam cparam;
  cparam.Init(impl->cfg, dmlc::parameter::kAllMatch);

  /* generate semantic model */
  impl->compiler.reset(Compiler::Create(impl->name, cparam));
  auto semantic_model = impl->compiler->Compile(*model_);
  if (verbose > 0) {
    LOG(INFO) << "Code generation finished. Writing code to files...";
  }

  /* write header */
  const std::string header_filename = dirpath_ + "/" + basename + ".h";
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
  std::vector<std::pair<std::string, size_t>> source_list;
  std::vector<std::string> object_list;
  if (semantic_model.units.size() == 1) {   // single file (translation unit)
    const std::string filename = basename + ".c";
    const std::string filename_full = dirpath_ + "/" + filename;
    const std::string objname = basename + ".o";
    if (verbose > 0) {
      LOG(INFO) << "Writing " << filename_full << " ...";
    }
    auto lines = semantic_model.units[0].Compile(header_filename);
    source_list.emplace_back(filename, lines.size());
    object_list.push_back(objname);
    common::WriteToFile(filename_full, lines);
  } else {  // multiple files (translation units)
    for (size_t i = 0; i < semantic_model.units.size(); ++i) {
      const std::string filename = basename + std::to_string(i) + ".c";
      const std::string filename_full = dirpath_ + "/" + filename;
      const std::string objname = basename + std::to_string(i) + ".o";
      if (verbose > 0) {
        LOG(INFO) << "Writing " << filename_full << " ...";
      }
      auto lines = semantic_model.units[i].Compile(header_filename);
      source_list.emplace_back(filename, lines.size());
      object_list.push_back(objname);
      common::WriteToFile(filename_full, lines);
    }
  }
  /* write build recipe, to be used by Python binding */
  {
    std::vector<std::pair<std::string, size_t>> sources;
    std::transform(source_list.begin(), source_list.end(),
      std::back_inserter(sources),
      [](const std::pair<std::string, size_t>& x) {
        return std::make_pair(x.first.substr(0, x.first.length() - 2),
                              x.second);
      });

    const std::string recipe_name = dirpath_ + "/recipe.json";
    if (verbose > 0) {
      LOG(INFO) << "Writing " << recipe_name << " ...";
    }
    std::unique_ptr<dmlc::Stream> fo(
                               dmlc::Stream::Create(recipe_name.c_str(), "w"));
    dmlc::ostream os(fo.get());
    auto writer = common::make_unique<dmlc::JSONWriter>(&os);
    writer->BeginObject();
    writer->WriteObjectKeyValue("target", basename);
    writer->WriteObjectKeyValue("sources", sources);
    writer->EndObject();
    // force flush before fo destruct.
    os.set_stream(nullptr);
  }
  /* write Makefile if on Linux */
#ifdef __linux__
  {
    std::string library_name = basename + ".so";
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
      oss << object_list[i] << ": " << source_list[i].first << std::endl
          << "\tgcc -c -O3 -o $@ $? -fPIC -std=c99 -flto" << std::endl;
    }
    oss << std::endl
        << "clean:" << std::endl
        << "\trm -fv " << library_name << " ";
    for (const auto& e : object_list) {
      oss << e << " ";
    }
    common::WriteToFile(dirpath_ + "/Makefile", {oss.str()});
    if (verbose > 0) {
      LOG(INFO) << "Writing " << dirpath_ + "/Makefile ...";
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

int TreeliteFreeModel(ModelHandle handle) {
  API_BEGIN();
  delete static_cast<Model*>(handle);
  API_END();
}

int TreeliteCreateTreeBuilder(TreeBuilderHandle* out) {
  API_BEGIN();
  auto builder = new frontend::TreeBuilder();
  *out = static_cast<TreeBuilderHandle>(builder);
  API_END();
}

int TreeliteDeleteTreeBuilder(TreeBuilderHandle handle) {
  API_BEGIN();
  delete static_cast<frontend::TreeBuilder*>(handle);
  API_END();
}

int TreeliteTreeBuilderCreateNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  return (builder->CreateNode(node_key)) ? 0 : -1;
  API_END();
}

int TreeliteTreeBuilderDeleteNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  return (builder->DeleteNode(node_key)) ? 0 : -1;
  API_END();
}

int TreeliteTreeBuilderSetRootNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  return (builder->SetRootNode(node_key)) ? 0 : -1;
  API_END();
}

int TreeliteTreeBuilderSetNumericalTestNode(TreeBuilderHandle handle,
                                            int node_key, unsigned feature_id,
                                            const char* opname,
                                            float threshold, int default_left,
                                            int left_child_key,
                                            int right_child_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK_GT(optable.count(opname), 0)
    << "No operator `" << opname << "\" exists";
  return (builder->SetNumericalTestNode(node_key, feature_id,
                                        optable.at(opname),
                                        static_cast<tl_float>(threshold),
                                        (default_left != 0),
                                        left_child_key, right_child_key)) \
                                        ? 0 : -1;
  API_END();
}

int TreeliteTreeBuilderSetCategoricalTestNode(
                                          TreeBuilderHandle handle,
                                          int node_key, unsigned feature_id,
                                          const unsigned char* left_categories,
                                          size_t left_categories_len,
                                          int default_left,
                                          int left_child_key,
                                          int right_child_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  std::vector<uint8_t> vec(left_categories_len);
  for (size_t i = 0; i < left_categories_len; ++i) {
    CHECK(left_categories[i] <= std::numeric_limits<uint8_t>::max());
    vec[i] = static_cast<uint8_t>(left_categories[i]);
  }
  return (builder->SetCategoricalTestNode(node_key, feature_id, vec,
                                          (default_left != 0),
                                          left_child_key, right_child_key)) \
                                          ? 0 : -1;
  API_END();
}

int TreeliteTreeBuilderSetLeafNode(TreeBuilderHandle handle, int node_key,
                                   float leaf_value) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  return (builder->SetLeafNode(node_key, static_cast<tl_float>(leaf_value))) \
                              ? 0 : -1;
  API_END();
}

int TreeliteTreeBuilderSetLeafVectorNode(TreeBuilderHandle handle,
                                         int node_key,
                                         const float* leaf_vector,
                                         size_t leaf_vector_len) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  std::vector<tl_float> vec(leaf_vector_len);
  for (size_t i = 0; i < leaf_vector_len; ++i) {
    vec[i] = static_cast<tl_float>(leaf_vector[i]);
  }
  return (builder->SetLeafVectorNode(node_key, vec)) ? 0 : -1;
  API_END();
}

int TreeliteCreateModelBuilder(int num_feature,
                               int num_output_group,
                               ModelBuilderHandle* out) {
  API_BEGIN();
  auto builder = new frontend::ModelBuilder(num_feature, num_output_group);
  *out = static_cast<ModelBuilderHandle>(builder);
  API_END();
}

int TreeliteModelBuilderSetModelParam(ModelBuilderHandle handle,
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

int TreeliteModelBuilderInsertTree(ModelBuilderHandle handle,
                                   TreeBuilderHandle tree_builder_handle,
                                   int index) {
  API_BEGIN();
  auto model_builder = static_cast<frontend::ModelBuilder*>(handle);
  auto tree_builder = static_cast<frontend::TreeBuilder*>(tree_builder_handle);
  return model_builder->InsertTree(tree_builder, index);
  API_END();
}

int TreeliteModelBuilderGetTree(ModelBuilderHandle handle, int index,
                                TreeBuilderHandle *out) {
  API_BEGIN();
  auto model_builder = static_cast<frontend::ModelBuilder*>(handle);
  auto tree_builder = &model_builder->GetTree(index);
  *out = static_cast<TreeBuilderHandle>(tree_builder);
  API_END();
}

int TreeliteModelBuilderDeleteTree(ModelBuilderHandle handle, int index) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  return (builder->DeleteTree(index)) ? 0 : -1;
  API_END();
}

int TreeliteModelBuilderCommitModel(ModelBuilderHandle handle,
                                    ModelHandle* out) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  Model* model = new Model();
  const bool result = builder->CommitModel(model);
  if (result) {
    *out = static_cast<ModelHandle>(model);
    return 0;
  } else {
    return -1;
  }
  API_END();
}
