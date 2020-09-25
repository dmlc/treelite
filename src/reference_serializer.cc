/*!
 * Copyright (c) 2020 by Contributors
 * \file reference_serializer.cc
 * \brief Reference serializer implementation
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>
#include <dmlc/io.h>
#include <dmlc/serializer.h>

namespace dmlc {
namespace serializer {

template <typename T>
struct Handler<treelite::ContiguousArray<T>> {
  inline static void Write(Stream* strm, const treelite::ContiguousArray<T>& data) {
    uint64_t sz = static_cast<uint64_t>(data.Size());
    strm->Write(sz);
    strm->Write(data.Data(), sz * sizeof(T));
  }

  inline static bool Read(Stream* strm, treelite::ContiguousArray<T>* data) {
    uint64_t sz;
    bool status = strm->Read(&sz);
    if (!status) {
      return false;
    }
    data->Resize(sz);
    return strm->Read(data->Data(), sz * sizeof(T));
  }
};

}  // namespace serializer
}  // namespace dmlc

namespace treelite {

template <typename ThresholdType, typename LeafOutputType>
void Tree<ThresholdType, LeafOutputType>::ReferenceSerialize(dmlc::Stream* fo) const {
  fo->Write(num_nodes);
  fo->Write(leaf_vector_);
  fo->Write(leaf_vector_offset_);
  fo->Write(left_categories_);
  fo->Write(left_categories_offset_);
  uint64_t sz = static_cast<uint64_t>(nodes_.Size());
  fo->Write(sz);
  fo->Write(nodes_.Data(), sz * sizeof(Tree::Node));

  // Sanity check
  CHECK_EQ(nodes_.Size(), num_nodes);
  CHECK_EQ(nodes_.Size() + 1, leaf_vector_offset_.Size());
  CHECK_EQ(leaf_vector_offset_.Back(), leaf_vector_.Size());
  CHECK_EQ(nodes_.Size() + 1, left_categories_offset_.Size());
  CHECK_EQ(left_categories_offset_.Back(), left_categories_.Size());
}

template <typename ThresholdType, typename LeafOutputType>
void ModelImpl<ThresholdType, LeafOutputType>::ReferenceSerialize(dmlc::Stream* fo) const {
  fo->Write(num_feature);
  fo->Write(num_output_group);
  fo->Write(random_forest_flag);
  fo->Write(&param, sizeof(param));
  uint64_t sz = static_cast<uint64_t>(trees.size());
  fo->Write(sz);
  for (const Tree<ThresholdType, LeafOutputType>& tree : trees) {
    tree.ReferenceSerialize(fo);
  }
}

template void Tree<float, uint32_t>::ReferenceSerialize(dmlc::Stream* fo) const;
template void Tree<float, float>::ReferenceSerialize(dmlc::Stream* fo) const;
template void Tree<double, uint32_t>::ReferenceSerialize(dmlc::Stream* fo) const;
template void Tree<double, double>::ReferenceSerialize(dmlc::Stream* fo) const;

template void ModelImpl<float, uint32_t>::ReferenceSerialize(dmlc::Stream* fo) const;
template void ModelImpl<float, float>::ReferenceSerialize(dmlc::Stream* fo) const;
template void ModelImpl<double, uint32_t>::ReferenceSerialize(dmlc::Stream* fo) const;
template void ModelImpl<double, double>::ReferenceSerialize(dmlc::Stream* fo) const;

}  // namespace treelite
