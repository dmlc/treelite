/*!
 * Copyright (c) 2023 by Contributors
 * \file json_importer.cc
 * \brief Function to construct a Treelite model from a JSON string
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <memory>

namespace treelite::frontend {

std::unique_ptr<treelite::Model> BuildModelFromJSONString(
    const char* json_str, const char* config_json) {
  return {};
}

}  // namespace treelite::frontend
