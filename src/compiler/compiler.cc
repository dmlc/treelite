/*!
 * Copyright 2017 by Contributors
 * \file compiler.cc
 * \brief Registry of compilers
 */
#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <dmlc/registry.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::treelite::CompilerReg);
}  // namespace dmlc

namespace treelite {
Compiler* Compiler::Create(const std::string& name,
                           const compiler::CompilerParam& param) {
  auto *e = ::dmlc::Registry< ::treelite::CompilerReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown compiler type " << name;
  }
  return (e->body)(param);
}
}  // namespace treelite

namespace treelite {
namespace compiler {
// register compiler parameter
DMLC_REGISTER_PARAMETER(CompilerParam);

// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(ast_native);
DMLC_REGISTRY_LINK_TAG(failsafe);
}  // namespace compiler
}  // namespace treelite
