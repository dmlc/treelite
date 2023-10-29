/*!
 * Copyright (c) 2023 by Contributors
 * \file field_accessor.cc
 * \author Hyunsu Cho
 * \brief C API for accessing fields in Treelite model
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/tree.h>

int TreeliteGetHeaderField(
    TreeliteModelHandle model, char const* name, TreelitePyBufferFrame* out_frame) {
  API_BEGIN();
  auto* model_ = static_cast<treelite::Model*>(model);
  *out_frame = model_->GetHeaderField(name);
  API_END();
}

int TreeliteGetTreeField(TreeliteModelHandle model, uint64_t tree_id, char const* name,
    TreelitePyBufferFrame* out_frame) {
  API_BEGIN();
  auto* model_ = static_cast<treelite::Model*>(model);
  *out_frame = model_->GetTreeField(tree_id, name);
  API_END();
}

int TreeliteSetHeaderField(
    TreeliteModelHandle model, char const* name, TreelitePyBufferFrame frame) {
  API_BEGIN();
  auto* model_ = static_cast<treelite::Model*>(model);
  model_->SetHeaderField(name, frame);
  API_END();
}

int TreeliteSetTreeField(
    TreeliteModelHandle model, uint64_t tree_id, char const* name, TreelitePyBufferFrame frame) {
  API_BEGIN();
  auto* model_ = static_cast<treelite::Model*>(model);
  model_->SetTreeField(tree_id, name, frame);
  API_END();
}
