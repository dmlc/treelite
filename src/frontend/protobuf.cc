/*!
 * Copyright 2017 by Contributors
 * \file protobuf.cc
 * \brief Frontend for protobuf model
 * \author Philip Cho
 */

#include <treelite/tree.h>

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(protobuf);

Model LoadProtobufModel(const char* filename) {
  LOG(FATAL) << "Not implemented yet";
  return Model();
}

}  // namespace frontend
}  // namespace treelite
