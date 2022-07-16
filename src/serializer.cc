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
  constexpr std::size_t kNumFrameInHeader = 5;
  if (frames.size() < kNumFrameInHeader) {
    throw Error(std::string("Insufficient number of frames: there must be at least ")
                + std::to_string(kNumFrameInHeader));
  }
  int idx = 0;
  auto header_primitive_field_handler = [&idx, &frames](auto* field) {
    InitScalarFromPyBuffer(field, frames[idx++]);
  };
  DeserializeTemplate(header_primitive_field_handler, threshold_type, leaf_output_type);
  if (idx != kNumFrameInHeader) {
    throw Error("Did not read from a sufficient number of frames");
  }

  std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
  model->InitFromPyBuffer(frames.begin() + kNumFrameInHeader, frames.end());
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
  DeserializeTemplate(header_primitive_field_handler, threshold_type, leaf_output_type);

  std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
  model->DeserializeFromFileImpl(src_fp);
  return model;
}

}  // namespace treelite
