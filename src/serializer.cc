/*!
 * Copyright (c) 2021 by Contributors
 * \file serializer.cc
 * \brief Implementation for serialization related functions
 * \author Hyunsu Cho
 */

#include <treelite/detail/serializer.h>
#include <treelite/error.h>
#include <treelite/tree.h>

#include <iostream>

namespace treelite {

std::vector<PyBufferFrame> Model::GetPyBuffer() {
  detail::serializer::PyBufferSerializerMixIn mixin{};
  detail::serializer::Serializer<detail::serializer::PyBufferSerializerMixIn> serializer{mixin};
  serializer.SerializeHeader(*this);
  serializer.SerializeTrees(*this);
  return mixin.GetFrames();
}

std::unique_ptr<Model> Model::CreateFromPyBuffer(std::vector<PyBufferFrame> frames) {
  detail::serializer::PyBufferDeserializerMixIn mixin{frames};
  detail::serializer::Deserializer<detail::serializer::PyBufferDeserializerMixIn> deserializer{
      mixin};
  std::unique_ptr<Model> model = deserializer.DeserializeHeaderAndCreateModel();
  deserializer.DeserializeTrees(*model);
  return model;
}

void Model::SerializeToStream(std::ostream& os) {
  detail::serializer::StreamSerializerMixIn mixin{os};
  detail::serializer::Serializer<detail::serializer::StreamSerializerMixIn> serializer{mixin};
  serializer.SerializeHeader(*this);
  serializer.SerializeTrees(*this);
}

std::unique_ptr<Model> Model::DeserializeFromStream(std::istream& is) {
  detail::serializer::StreamDeserializerMixIn mixin{is};
  detail::serializer::Deserializer<detail::serializer::StreamDeserializerMixIn> deserializer{mixin};
  std::unique_ptr<Model> model = deserializer.DeserializeHeaderAndCreateModel();
  deserializer.DeserializeTrees(*model);
  return model;
}

}  // namespace treelite
