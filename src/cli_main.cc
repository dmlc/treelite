/*!
 * Copyright 2017 by Contributors
 * \file cli_main.cc
 * \brief The command line interface program of treelite.
 * \author Philip Cho
 */

#include <treelite/frontend.h>
#include <treelite/annotator.h>
#include <treelite/compiler.h>
#include <treelite/semantic.h>
#include <dmlc/config.h>
#include <dmlc/data.h>
#include <fstream>
#include <memory>
#include <vector>
#include <queue>
#include <iterator>
#include <string>
#include <omp.h>
#include "./compiler/param.h"

namespace treelite {

enum CLITask {
  kCodegen = 0,
  kAnnotate = 1
};

enum InputFormat {
  kLibSVM = 0,
  kCSV = 1,
  kLibFM = 2,
};

enum ModelFormat {
  kXGBModel = 0,
  kLGBModel = 1
};

inline const char* FileFormatString(int format) {
  switch (format) {
    case kLibSVM: return "libsvm";
    case kCSV: return "csv";
    case kLibFM: return "libfm";
  }
  return "";
}

struct CLIParam : public dmlc::Parameter<CLIParam> {
  /*! \brief the task name */
  int task;
  /*! \brief whether verbose */
  int verbose;
  /*! \brief model format */
  int format;
  /*! \brief model file */
  std::string model_in;
  /*! \brief generated code file */
  std::string name_codegen;
  /*! \brief name of generated annotation file */
  std::string name_annotate;
  /*! \brief the path of training set -- used for annotation */
  std::string train_path;
  /*! \brief training set file format */
  int train_format;
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  int nthread;
  /*! \brief all the configurations */
  std::vector<std::pair<std::string, std::string> > cfg;

  // declare parameters
  DMLC_DECLARE_PARAMETER(CLIParam) {
    DMLC_DECLARE_FIELD(task).set_default(kCodegen)
        .add_enum("train", kCodegen)
        .add_enum("annotate", kAnnotate)
        .describe("Task to be performed by the CLI program.");
    DMLC_DECLARE_FIELD(verbose).set_default(0)
        .describe("Produce extra messages if >0");
    DMLC_DECLARE_FIELD(format)
        .add_enum("xgboost", kXGBModel)
        .add_enum("lightgbm", kLGBModel)
        .describe("Model format");
    DMLC_DECLARE_FIELD(model_in).describe("Input model path");
    DMLC_DECLARE_FIELD(name_codegen).set_default("dump.c")
        .describe("generated code file");
    DMLC_DECLARE_FIELD(name_annotate).set_default("annotate.json")
        .describe("Name of generated annotation file");
    DMLC_DECLARE_FIELD(train_path).set_default("NULL")
        .describe("Training data path; used for annotation");
    DMLC_DECLARE_FIELD(train_format).set_default(kLibSVM)
        .add_enum("libsvm", kLibSVM)
        .add_enum("csv", kCSV)
        .add_enum("libfm", kLibFM);
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe(
        "Number of threads to use.");

    // alias
    DMLC_DECLARE_ALIAS(train_path, data);
    DMLC_DECLARE_ALIAS(train_format, data_format);
  }
  // customized configure function of CLIParam
  inline void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) {
    this->cfg = cfg;
    this->InitAllowUnknown(cfg);
  }
};

DMLC_REGISTER_PARAMETER(CLIParam);

Model ParseModel(const CLIParam& param) {
  switch (param.format) {
   case kXGBModel:
    return frontend::LoadXGBoostModel(param.model_in.c_str());
   case kLGBModel:
    return frontend::LoadLightGBMModel(param.model_in.c_str());
   default:
    LOG(FATAL) << "Unknown model format";
    return {};  // avoid compiler warning
  }
}

void CLICodegen(const CLIParam& param) {
  compiler::CompilerParam cparam;
  cparam.InitAllowUnknown(param.cfg);

  Model model = ParseModel(param);
  LOG(INFO) << "model size = " << model.trees.size();

  std::unique_ptr<Compiler> compiler(Compiler::Create("recursive", cparam));
  auto semantic_model = compiler->Compile(model);
  /* write header */
  const std::string header_filename = param.name_codegen + ".h";
  {
    std::vector<std::string> lines;
    common::TransformPushBack(&lines, semantic_model.common_header->Compile(),
      [] (std::string line) {
        return line;
      });
    lines.emplace_back();
    std::ostringstream oss;
    std::copy(semantic_model.function_registry.begin(),
              semantic_model.function_registry.end(),
              std::ostream_iterator<std::string>(oss, ";\n"));
    lines.push_back(oss.str());
    common::WriteToFile(header_filename, lines);
  }
  /* write source file(s) */
  std::vector<std::string> source_list;
  std::vector<std::string> object_list;
  if (semantic_model.units.size() == 1) {   // single file (translation unit)
    const std::string filename = param.name_codegen + ".c";
    const std::string objname = param.name_codegen + ".o";
    source_list.push_back(common::GetBasename(filename));
    object_list.push_back(common::GetBasename(objname));
    auto lines = semantic_model.units[0].Compile(header_filename);
    common::WriteToFile(filename, lines);
  } else {  // multiple files (translation units)
    for (size_t i = 0; i < semantic_model.units.size(); ++i) {
      const std::string filename = param.name_codegen + std::to_string(i)+ ".c";
      const std::string objname = param.name_codegen + std::to_string(i) + ".o";
      source_list.push_back(common::GetBasename(filename));
      object_list.push_back(common::GetBasename(objname));
      auto lines = semantic_model.units[i].Compile(header_filename);
      common::WriteToFile(filename, lines);
    }
  }
  /* write Makefile */
  {
    const std::string library_name
      = common::GetBasename(param.name_codegen + ".so");
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
    common::WriteToFile(param.name_codegen + ".Makefile", {oss.str()});
  }
}

void CLIAnnotate(const CLIParam& param) {
  Model model = ParseModel(param);
  LOG(INFO) << "model size = " << model.trees.size();

  CHECK_NE(param.train_path, "NULL")
    << "Need to specify train_path paramter for annotation task";
  std::unique_ptr<DMatrix> dmat(DMatrix::Create(param.train_path.c_str(),
                                         FileFormatString(param.train_format),
                                         param.nthread, param.verbose));
  BranchAnnotator annotator;
  annotator.Annotate(model, dmat.get(), param.nthread, param.verbose);
  // write to json file
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(
                                   param.name_annotate.c_str(), "w"));
  annotator.Save(fo.get());
}

int CLIRunTask(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: <config>\n");
    return 0;
  }

  std::vector<std::pair<std::string, std::string> > cfg;

  std::ifstream cfgfile(argv[1], std::ifstream::in);
  dmlc::Config itr(cfgfile);
  for (const auto& entry : itr) {
    cfg.push_back(std::make_pair(entry.first, entry.second));
  }
  cfgfile.close();

  for (int i = 2; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      cfg.push_back(std::make_pair(std::string(name), std::string(val)));
    }
  }

  CLIParam param;
  param.Configure(cfg);

  switch (param.task) {
    case kCodegen: CLICodegen(param); break;
    case kAnnotate: CLIAnnotate(param); break;
  }

  return 0;
}

}  // namespace treelite

int main(int argc, char* argv[]) {
  return treelite::CLIRunTask(argc, argv);
}
