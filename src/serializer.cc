/*!
 * Copyright (c) 2021 by Contributors
 * \file serializer.cc
 * \brief Implementation for serialization related functions
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>
#include <treelite/error.h>

namespace treelite {

std::vector<PyBufferFrame>
Model::GetPyBuffer() {
  std::vector<PyBufferFrame> buffer;
  auto header_primitive_field_handler = [&buffer](auto* field) {
    buffer.push_back(GetPyBufferFromScalar(field));
  };
  SerializeTemplate(header_primitive_field_handler);
  this->GetPyBuffer(&buffer);
  return buffer;
}

std::unique_ptr<Model>
Model::CreateFromPyBuffer(std::vector<PyBufferFrame> frames) {
  TypeInfo threshold_type, leaf_output_type;
  auto it = frames.begin();
  auto header_primitive_field_handler = [&it](auto* field) {
    InitScalarFromPyBuffer(field, *it++);
  };
  int32_t major_ver;
  int32_t minor_ver;
  int32_t patch_ver;
  DeserializeTemplate(header_primitive_field_handler, major_ver, minor_ver, patch_ver,
    threshold_type, leaf_output_type);

  std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
  model->major_ver_ = major_ver;
  model->minor_ver_ = minor_ver;
  model->patch_ver_ = patch_ver;
  std::size_t num_frame = std::distance(it, frames.end());
  model->InitFromPyBuffer(it, num_frame);
  return model;
}

void
Model::SerializeToFile(FILE* dest_fp) {
  auto header_primitive_field_handler = [dest_fp](auto* field) {
    WriteScalarToFile(field, dest_fp);
  };
  SerializeTemplate(header_primitive_field_handler);
  this->SerializeToFileImpl(dest_fp);
}

std::unique_ptr<Model>
Model::DeserializeFromFile(FILE* src_fp) {
  TypeInfo threshold_type, leaf_output_type;
  int idx = 0;
  auto header_primitive_field_handler = [src_fp](auto* field) {
    ReadScalarFromFile(field, src_fp);
  };
  int32_t major_ver;
  int32_t minor_ver;
  int32_t patch_ver;
  DeserializeTemplate(header_primitive_field_handler, major_ver, minor_ver, patch_ver,
    threshold_type, leaf_output_type);

  std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
  model->major_ver_ = major_ver;
  model->minor_ver_ = minor_ver;
  model->patch_ver_ = patch_ver;
  model->DeserializeFromFileImpl(src_fp);
  return model;
}

}  // namespace treelite
