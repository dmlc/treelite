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
#include <treelite/predictor.h>
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
#include "./common/filesystem.h"

namespace treelite {

enum CLITask {
  kCodegen = 0,
  kAnnotate = 1,
  kPredict = 2
};

enum InputFormat {
  kLibSVM = 0,
  kCSV = 1,
  kLibFM = 2
};

enum ModelFormat {
  kXGBModel = 0,
  kLGBModel = 1,
  kProtobuf = 2
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
  /*! \brief directory name for generated code files */
  std::string name_codegen_dir;
  /*! \brief name of generated annotation file */
  std::string name_annotate;
  /*! \brief name of text file to save prediction */
  std::string name_pred;
  /*! \brief the path of training set: used for annotation */
  std::string train_data_path;
  /*! \brief the path of test set: used for prediction */
  std::string test_data_path;
  /*! \brief the path of compiled dynamic shared library: used for prediction */
  std::string codelib_path;
  /*! \brief training set file format */
  int train_format;
  /*! \brief test set file format */
  int test_format;
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  int nthread;
  /*! \brief whether to predict margin instead of
             transformed probability */
  int pred_margin;
  /*! \brief all the configurations */
  std::vector<std::pair<std::string, std::string> > cfg;

  // declare parameters
  DMLC_DECLARE_PARAMETER(CLIParam) {
    DMLC_DECLARE_FIELD(task).set_default(kCodegen)
        .add_enum("train", kCodegen)
        .add_enum("annotate", kAnnotate)
        .add_enum("predict", kPredict)
        .describe("Task to be performed by the CLI program.");
    DMLC_DECLARE_FIELD(verbose).set_default(0)
        .describe("Produce extra messages if >0");
    DMLC_DECLARE_FIELD(format)
        .add_enum("xgboost", kXGBModel)
        .add_enum("lightgbm", kLGBModel)
        .add_enum("protobuf", kProtobuf)
        .describe("Model format");
    DMLC_DECLARE_FIELD(model_in).set_default("NULL")
        .describe("Input model path");
    DMLC_DECLARE_FIELD(name_codegen_dir).set_default("codegen")
        .describe("directory name for generated code files");
    DMLC_DECLARE_FIELD(name_annotate).set_default("annotate.json")
        .describe("Name of generated annotation file");
    DMLC_DECLARE_FIELD(name_pred).set_default("pred.txt")
        .describe("Name of text file to save prediction");
    DMLC_DECLARE_FIELD(train_data_path).set_default("NULL")
        .describe("Training data path; used for annotation");
    DMLC_DECLARE_FIELD(test_data_path).set_default("NULL")
        .describe("Test data path; used prediction");
    DMLC_DECLARE_FIELD(codelib_path).set_default("NULL")
        .describe("Path to compiled dynamic shared library (.so/.dll/.dylib); "
                  "used for prediction");
    DMLC_DECLARE_FIELD(train_format).set_default(kLibSVM)
        .add_enum("libsvm", kLibSVM)
        .add_enum("csv", kCSV)
        .add_enum("libfm", kLibFM)
        .describe("training set data format");
    DMLC_DECLARE_FIELD(test_format).set_default(kLibSVM)
        .add_enum("libsvm", kLibSVM)
        .add_enum("csv", kCSV)
        .add_enum("libfm", kLibFM)
        .describe("test set data format");
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe(
        "Number of threads to use.");
    DMLC_DECLARE_FIELD(pred_margin).set_default(0).describe(
        "if >0, predict margin instead of transformed probability");

    // alias
    DMLC_DECLARE_ALIAS(train_data_path, data);
    DMLC_DECLARE_ALIAS(test_data_path, test:data);
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
  CHECK(param.model_in != "NULL") << "model_in parameter must be provided";
  switch (param.format) {
   case kXGBModel:
    return frontend::LoadXGBoostModel(param.model_in.c_str());
   case kLGBModel:
    return frontend::LoadLightGBMModel(param.model_in.c_str());
   case kProtobuf:
    return frontend::LoadProtobufModel(param.model_in.c_str());
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

  // create directory named name_codegen_dir
  common::filesystem::CreateDirectoryIfNotExist(param.name_codegen_dir.c_str());
  const std::string basename
    = common::filesystem::GetBasename(param.name_codegen_dir);

  std::unique_ptr<Compiler> compiler(Compiler::Create("recursive", cparam));
  auto semantic_model = compiler->Compile(model);
  /* write header */
  const std::string header_filename
    = param.name_codegen_dir + "/" + basename + ".h";
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
    const std::string filename = param.name_codegen_dir + "/" + basename + ".c";
    const std::string objname = param.name_codegen_dir + "/" + basename + ".o";
    source_list.push_back(common::filesystem::GetBasename(filename));
    object_list.push_back(common::filesystem::GetBasename(objname));
    auto lines = semantic_model.units[0].Compile(header_filename);
    common::WriteToFile(filename, lines);
  } else {  // multiple files (translation units)
    for (size_t i = 0; i < semantic_model.units.size(); ++i) {
      const std::string filename
        = param.name_codegen_dir + "/" + basename + std::to_string(i) + ".c";
      const std::string objname
        = param.name_codegen_dir + "/" + basename + std::to_string(i) + ".o";
      source_list.push_back(common::filesystem::GetBasename(filename));
      object_list.push_back(common::filesystem::GetBasename(objname));
      auto lines = semantic_model.units[i].Compile(header_filename);
      common::WriteToFile(filename, lines);
    }
  }
  /* write Makefile */
#ifdef __linux__
  {
    const std::string library_name
      = common::filesystem::GetBasename(param.name_codegen_dir + "/"
                                        + basename + ".so");
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
    common::WriteToFile(param.name_codegen_dir + "/Makefile",
                        {oss.str()});
  }
#endif
}

void CLIAnnotate(const CLIParam& param) {
  Model model = ParseModel(param);
  LOG(INFO) << "model size = " << model.trees.size();

  CHECK_NE(param.train_data_path, "NULL")
    << "Need to specify train_data_path paramter for annotation task";
  std::unique_ptr<DMatrix> dmat(DMatrix::Create(param.train_data_path.c_str(),
                                         FileFormatString(param.train_format),
                                         param.nthread, param.verbose));
  BranchAnnotator annotator;
  annotator.Annotate(model, dmat.get(), param.nthread, param.verbose);
  // write to json file
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(
                                   param.name_annotate.c_str(), "w"));
  annotator.Save(fo.get());
}

void CLIPredict(const CLIParam& param) {

  CHECK_NE(param.codelib_path, "NULL")
    << "Need to specify codelib_path paramter for prediction task";
  CHECK_NE(param.test_data_path, "NULL")
    << "Need to specify test_data_path paramter for prediction task";
  std::unique_ptr<DMatrix> dmat(DMatrix::Create(param.test_data_path.c_str(),
                                         FileFormatString(param.test_format),
                                         param.nthread, param.verbose));
  Predictor predictor;
  predictor.Load(param.codelib_path.c_str());
  size_t result_size = predictor.QueryResultSize(dmat.get());
  std::vector<float> result(result_size);
  if (param.pred_margin > 0) {
    predictor.PredictRaw(dmat.get(), param.nthread, param.verbose, &result[0]);
  } else {
    result_size
     = predictor.Predict(dmat.get(), param.nthread, param.verbose, &result[0]);
  }
  // write to text file
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(
                                   param.name_pred.c_str(), "w"));
  dmlc::ostream os(fo.get());
  for (size_t i = 0; i < result_size; ++i) {
    os << result[i] << std::endl;
  }
  // force flush before fo destruct.
  os.set_stream(nullptr);
}

int CLIRunTask(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: <config>\n");
    return 0;
  }

  std::vector<std::pair<std::string, std::string> > cfg;

  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(argv[1], "r"));
    dmlc::istream cfgfile(fi.get());
    dmlc::Config itr(cfgfile);
    for (const auto& entry : itr) {
      cfg.push_back(std::make_pair(entry.first, entry.second));
    }
  }

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
    case kPredict: CLIPredict(param); break;
  }

  return 0;
}

}  // namespace treelite

int main(int argc, char* argv[]) {
  return treelite::CLIRunTask(argc, argv);
}
