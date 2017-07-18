/*!
 * Copyright 2017 by Contributors
 * \file compiler.cc
 * \brief Registry of compilers
 */
#include <treelite/compiler.h>
#include <dmlc/registry.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::treelite::CompilerReg);
}  // namespace dmlc

namespace treelite {
Compiler* Compiler::Create(const std::string& name) {
  auto *e = ::dmlc::Registry< ::treelite::CompilerReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown compiler type " << name;
  }
  return (e->body)();
}
}  // namespace treelite

namespace treelite {
namespace compiler {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(simple);
}  // namespace compiler
}  // namespace treelite
