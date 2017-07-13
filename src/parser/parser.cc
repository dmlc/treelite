/*!
 * Copyright 2017 by Contributors
 * \file parser.cc
 * \brief Registry of parsers.
 */
#include <treelite/parser.h>
#include <dmlc/registry.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::treelite::ParserReg);
}  // namespace dmlc

namespace treelite {
Parser* Parser::Create(const std::string& name) {
  auto *e = ::dmlc::Registry< ::treelite::ParserReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown parser type " << name;
  }
  return (e->body)();
}
}  // namespace treelite

namespace treelite {
namespace parser {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(xgboost);
DMLC_REGISTRY_LINK_TAG(lightgbm);
}  // namespace parser
}  // namespace treelite
