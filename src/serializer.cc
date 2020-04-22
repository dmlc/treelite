/*!
 * Copyright 2020 by Contributors
 * \file serializer.cc
 * \brief Serialization of Tree and Model objects
 * \author Philip Cho
 */

#include <treelite/tree.h>
#include <dmlc/serializer.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>

namespace treelite {

void Tree::Serialize(dmlc::Stream* fo) const {
  fo->Write(num_nodes);
  fo->Write(leaf_vector_);
  fo->Write(left_categories_);
  uint64_t sz = static_cast<uint64_t>(nodes_.size());
  fo->Write(sz);
  fo->Write(nodes_.data(), sz * sizeof(Tree::Node));
  CHECK_EQ(nodes_.size(), left_categories_.size());
}

void Tree::Deserialize(dmlc::Stream* fi) {
  fi->Read(&num_nodes);
  fi->Read(&leaf_vector_);
  fi->Read(&left_categories_);
  uint64_t sz = 0;
  fi->Read(&sz);
  nodes_.clear();
  nodes_.resize(sz, Node(nullptr, nullptr));
  fi->Read(nodes_.data(), sz * sizeof(Tree::Node));
  for (uint64_t i = 0; i < sz; ++i) {
    nodes_[i].leaf_vector_ = &leaf_vector_[i];
    nodes_[i].left_categories_ = &left_categories_[i];
  }
}

void Model::Serialize(dmlc::Stream* fo) const {
  fo->Write(num_feature);
  fo->Write(num_output_group);
  fo->Write(random_forest_flag);
  fo->Write(&param, sizeof(param));
  uint64_t sz = static_cast<uint64_t>(trees.size());
  fo->Write(sz);
  for (const Tree& tree : trees) {
    tree.Serialize(fo);
  }
}

void Model::Deserialize(dmlc::Stream* fi) {
  fi->Read(&num_feature);
  fi->Read(&num_output_group);
  fi->Read(&random_forest_flag);
  fi->Read(&param, sizeof(param));
  uint64_t sz = 0;
  fi->Read(&sz);
  for (uint64_t i = 0; i < sz; ++i) {
    Tree tree;
    tree.Deserialize(fi);
    trees.push_back(std::move(tree));
  }
}

}  // namespace treelite
