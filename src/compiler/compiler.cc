/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file compiler.cc
 * \brief Registry of compilers
 */
#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <treelite/logging.h>
#include <rapidjson/document.h>
#include <limits>
#include "./ast_native.h"
#include "./failsafe.h"

namespace treelite {

Compiler* Compiler::Create(const std::string& name, const char* param_json_str) {
  compiler::CompilerParam param = compiler::CompilerParam::ParseFromJSON(param_json_str);
  if (name == "ast_native") {
    return new compiler::ASTNativeCompiler(param);
  } else if (name == "failsafe") {
    return new compiler::FailSafeCompiler(param);
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized compiler '" << name << "'";
    return nullptr;
  }
}

namespace compiler {

CompilerParam
CompilerParam::ParseFromJSON(const char* param_json_str) {
  CompilerParam param;
  // Default values
  param.annotate_in = "NULL";
  param.quantize = 0;
  param.parallel_comp = 0;
  param.verbose = 0;
  param.native_lib_name = "predictor";
  param.code_folding_req = std::numeric_limits<double>::infinity();
  param.dump_array_as_elf = 0;

  rapidjson::Document doc;
  doc.Parse(param_json_str);
  TREELITE_CHECK(doc.IsObject()) << "Got an invalid JSON string:\n" << param_json_str;
  for (const auto& e : doc.GetObject()) {
    const std::string key = e.name.GetString();
    if (key == "annotate_in") {
      TREELITE_CHECK(e.value.IsString()) << "Expected a string for 'annotate_in'";
      param.annotate_in = e.value.GetString();
    } else if (key == "quantize") {
      TREELITE_CHECK(e.value.IsInt()) << "Expected an integer for 'quantize'";
      param.quantize = e.value.GetInt();
      TREELITE_CHECK_GE(param.quantize, 0) << "'quantize' must be 0 or greater";
    } else if (key == "parallel_comp") {
      TREELITE_CHECK(e.value.IsInt()) << "Expected an integer for 'parallel_comp'";
      param.parallel_comp = e.value.GetInt();
      TREELITE_CHECK_GE(param.parallel_comp, 0) << "'parallel_comp' must be 0 or greater";
    } else if (key == "verbose") {
      TREELITE_CHECK(e.value.IsInt()) << "Expected an integer for 'verbose'";
      param.verbose = e.value.GetInt();
    } else if (key == "native_lib_name") {
      TREELITE_CHECK(e.value.IsString()) << "Expected a string for 'native_lib_name'";
      param.native_lib_name = e.value.GetString();
    } else if (key == "code_folding_req") {
      TREELITE_CHECK(e.value.IsDouble())
          << "Expected a floating-point decimal for 'code_folding_req'";
      param.code_folding_req = e.value.GetDouble();
      TREELITE_CHECK_GE(param.code_folding_req, 0) << "'code_folding_req' must be 0 or greater";
    } else if (key == "dump_array_as_elf") {
      TREELITE_CHECK(e.value.IsInt()) << "Expected an integer for 'dump_array_as_elf'";
      param.dump_array_as_elf = e.value.GetInt();
      TREELITE_CHECK_GE(param.dump_array_as_elf, 0) << "'dump_array_as_elf' must be 0 or greater";
    } else {
      TREELITE_LOG(FATAL) << "Unrecognized key '" << key << "' in JSON";
    }
  }

  return param;
}

}  // namespace compiler

}  // namespace treelite
