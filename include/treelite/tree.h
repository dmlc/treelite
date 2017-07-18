/*!
 * Copyright 2017 by Contributors
 * \file tree.h
 * \brief model structure for tree
 * \author Philip Cho
 */
#ifndef TREELITE_TREE_H_
#define TREELITE_TREE_H_

#include <treelite/base.h>
#include <treelite/common.h>
#include <dmlc/logging.h>
#include <vector>
#include <limits>

namespace treelite {

class Tree {
 public:
  /*! \brief comparison operators */
  enum class Operator : int8_t {
    kEQ, kLT, kLE, kGT, kGE  // ==, <, <=, >, >=
  };

  /*! \brief tree node */
  class Node {
   public:
    Node() : sindex_(0) {}
    /*! \brief index of left child */
    inline int cleft() const {
      return this->cleft_;
    }
    /*! \brief index of right child */
    inline int cright() const {
      return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    inline int cdefault() const {
      return this->default_left() ? this->cleft() : this->cright();
    }
    /*! \brief feature index of split condition */
    inline unsigned split_index() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    inline bool default_left() const {
      return (sindex_ >> 31) != 0;
    }
    /*! \brief whether current node is leaf node */
    inline bool is_leaf() const {
      return cleft_ == -1;
    }
    /*! \return get leaf value of leaf node */
    inline tl_float leaf_value() const {
      return (this->info_).leaf_value;
    }
    /*! \return get threshold of the node */
    inline tl_float threshold() const {
      return (this->info_).threshold;
    }
    /*! \brief get parent of the node */
    inline int parent() const {
      return parent_ & ((1U << 31) - 1);
    }
    /*! \brief whether current node is left child */
    inline bool is_left_child() const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether current node is root */
    inline bool is_root() const {
      return parent_ == -1;
    }
    /*! \brief get comparison operator */
    inline Operator comparison_op() const {
      return cmp_;
    }
    /*!
     * \brief set split condition of current node
     * \param split_index feature index to split
     * \param threshold threshold value
     * \param default_left the default direction when feature is unknown
     * \param cmp comparison operator to compare between feature value and threshold
     */
    inline void set_split(unsigned split_index, tl_float threshold,
                          bool default_left, Operator cmp) {
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).threshold = threshold;
      this->cmp_ = cmp;
    }
    /*!
     * \brief set the leaf value of the node
     * \param value leaf value
     */
    inline void set_leaf(tl_float value) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = -1;
    }
    // set parent
    inline void set_parent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }

   private:
    friend class Tree;
    /*!
     * \brief in leaf node, we have weights, in non-leaf nodes,
     *        we have threshold
     */
    union Info {
      tl_float leaf_value;
      tl_float threshold;
    };
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    int parent_;
    // pointer to left and right children
    int cleft_, cright_;
    // split feature index;
    // highest bit indicates default direction for missing values
    unsigned sindex_;
    // extra info: weights or threshold
    Info info_;
    // operator to use for expression of form [fval] OP [threshold]
    // If the expression evaluates to true, take the left child;
    // otherwise, take the right child.
    Operator cmp_;
  };

 private:
  // vector of nodes
  std::vector<Node> nodes;
  // allocate a new node
  inline int AllocNode() {
    int nd = num_nodes++;
    CHECK_LT(num_nodes, std::numeric_limits<int>::max())
        << "number of nodes in the tree exceed 2^31";
    nodes.resize(num_nodes);
    return nd;
  }

 public:
  int num_nodes;
  /*! \brief get node given nid */
  inline Node& operator[](int nid) {
    return nodes[nid];
  }
  /*! \brief get node given nid */
  inline const Node& operator[](int nid) const {
    return nodes[nid];
  }
  /*! \brief initialize the model with a single root node */
  inline void Init() {
    num_nodes = 1;
    nodes.resize(1);
    nodes[0].set_leaf(0.0f);
    nodes[0].set_parent(-1);
  }
  /*!
   * \brief add child nodes to node
   * \param nid node id to add children to
   */
  inline void AddChilds(int nid) {
    const int cleft = this->AllocNode();
    const int cright = this->AllocNode();
    nodes[nid].cleft_ = cleft;
    nodes[nid].cright_ = cright;
    nodes[cleft].set_parent(nid, true);
    nodes[cright].set_parent(nid, false);
  }
};

// thin wrapper for ensemble model
struct Model {
  std::vector<Tree> trees;
  int num_features;
};

}  // namespace treelite
#endif  // TREELITE_TREE_H_
