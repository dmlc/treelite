/*!
 * Copyright 2017 by Contributors
 * \file cli_main.cc
 * \brief The command line interface program of treelite.
 * \author Philip Cho
 */

#include <treelite/compiler.h>
#include <treelite/parser.h>
#include <treelite/semantic.h>
#include <dmlc/config.h>
#include <dmlc/data.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <queue>
#include <iterator>
#include <string>
#include <omp.h>
#include "./compiler/param.h"
#include "./traversal.h"

namespace treelite {

enum CLITask {
  kCodegen = 0,
  kAnnotate = 1
};

enum FileFormat {
  kLibSVM = 0,
  kCSV = 1,
  kLibFM = 2
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
  /*! \brief whether silent */
  int silent;
  /*! \brief model format */
  std::string format;
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
    DMLC_DECLARE_FIELD(silent).set_default(0).set_range(0, 2)
        .describe("Silence level during the task; >0 generates more messages");
    DMLC_DECLARE_FIELD(format).describe("Model format");
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

void CLICodegen(const CLIParam& param) {
  compiler::CompilerParam cparam;
  cparam.InitAllowUnknown(param.cfg);

  std::unique_ptr<Parser> parser(Parser::Create(param.format));
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(
                                   param.model_in.c_str(), "r"));
  parser->Load(fi.get());

  Model model = parser->Export();
  LOG(INFO) << "model size = " << model.trees.size();

  std::unique_ptr<Compiler> compiler(Compiler::Create("recursive", cparam));
  auto semantic_model = compiler->Export(model);

  std::ostringstream oss;
  const auto& reg = semantic::FunctionEntry::GetRegistry();
  std::copy(reg.begin(), reg.end(),
            std::ostream_iterator<std::string>(oss, "\n"));
  std::cerr << "FunctionEntry = \n" << oss.str();

  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(
                                   param.name_codegen.c_str(), "w"));
  dmlc::ostream os(fo.get());
  auto lines = semantic_model->Compile();
  std::copy(lines.begin(), lines.end(),
            std::ostream_iterator<std::string>(os, "\n"));
}

void CLIAnnotate(const CLIParam& param) {
  std::unique_ptr<Parser> model_parser(Parser::Create(param.format));
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(
                                   param.model_in.c_str(), "r"));
  model_parser->Load(fi.get());
  
  Model model = model_parser->Export();
  LOG(INFO) << "model size = " << model.trees.size();

  CHECK_NE(param.train_path, "NULL")
    << "Need to specify train_path paramter for annotation task";
  std::unique_ptr<dmlc::Parser<uint32_t> > data_parser(
      dmlc::Parser<uint32_t>::Create(param.train_path.c_str(), 0, 1,
                                     FileFormatString(param.train_format)));
  std::vector<size_t> counts;
  std::vector<uint32_t> row_ptr;
  common::ComputeBranchFrequenciesFromData(model, data_parser.get(), &counts,
                                         &row_ptr, param.nthread, param.silent);
  // write to json file
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(
                                   param.name_annotate.c_str(), "w"));
  dmlc::ostream os(fo.get());
  auto writer = common::make_unique<dmlc::JSONWriter>(&os);
  writer->BeginArray();
  for (size_t tree_id = 0; tree_id < model.trees.size(); ++tree_id) {
    writer->WriteArraySeperator();
    writer->BeginArray(false);
    const uint32_t ibegin = row_ptr[tree_id];
    const uint32_t iend = row_ptr[tree_id + 1];
    for (uint32_t i = ibegin; i < iend; ++i) {
      writer->WriteArrayItem(counts[i]);
    }
    writer->EndArray();
  }
  writer->EndArray();
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
