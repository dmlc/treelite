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

/*! \brief in-memory representation of a decision tree */
class Tree {
 public:
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
    /*!
     * \return get leaf vector of leaf node; useful for multi-class
     * random forest classifier
     */
    inline const std::vector<tl_float>& leaf_vector() const {
      return this->leaf_vector_;
    }
    /*!
     * \return tests whether leaf node has a non-empty leaf vector
     */
    inline bool has_leaf_vector() const {
      return !(this->leaf_vector_.empty());
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
    /*! \brief get categories for left child node */
    inline const std::vector<uint8_t>& left_categories() const {
      return left_categories_;
    }
    /*! \briet get feature split type */
    inline SplitFeatureType split_type() const {
      return split_type_;
    }
    /*!
     * \brief create a numerical split
     * \param split_index feature index to split
     * \param threshold threshold value
     * \param default_left the default direction when feature is unknown
     * \param cmp comparison operator to compare between feature value and
     *            threshold
     */
    inline void set_numerical_split(unsigned split_index, tl_float threshold,
                                    bool default_left, Operator cmp) {
      CHECK_LT(split_index, (1U << 31) - 1) << "split_index too big";
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).threshold = threshold;
      this->cmp_ = cmp;
      this->split_type_ = SplitFeatureType::kNumerical;
    }
    /*!
     * \brief create a categorical split
     * \param split_index feature index to split
     * \param threshold threshold value
     * \param default_left the default direction when feature is unknown
     * \param cmp comparison operator to compare between feature value and
     *            threshold
     */
    inline void set_categorical_split(unsigned split_index, bool default_left,
                                 const std::vector<uint8_t>& left_categories) {
      CHECK_LT(split_index, (1U << 31) - 1) << "split_index too big";
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      this->left_categories_ = left_categories;
      std::sort(this->left_categories_.begin(), this->left_categories_.end());
      this->split_type_ = SplitFeatureType::kCategorical;
    }
    /*!
     * \brief set the leaf value of the node
     * \param value leaf value
     */
    inline void set_leaf(tl_float value) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = -1;
      this->split_type_ = SplitFeatureType::kNone;
    }
    /*!
     * \brief set the leaf vector of the node; useful for multi-class
     * random forest classifier
     * \param leaf_vector leaf vector
     */
    inline void set_leaf_vector(const std::vector<tl_float>& leaf_vector) {
      this->leaf_vector_ = leaf_vector;
      this->cleft_ = -1;
      this->cright_ = -1;
      this->split_type_ = SplitFeatureType::kNone;
    }
    /*!
     * \brief set parent of the node
     * \param pidx node id of the parent
     * \param is_left_child whether the node is left child or not
     */
    inline void set_parent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }

   private:
    friend class Tree;
    /*! \brief store either leaf value or decision threshold */
    union Info {
      tl_float leaf_value;  // for leaf nodes
      tl_float threshold;   // for non-leaf nodes
    };
    /*!
     * \brief leaf vector: only used for random forests with
     *                     multi-class classification
     */
    std::vector<tl_float> leaf_vector_;
    /*!
     * \brief pointer to parent
     * highest bit is used to indicate whether it's a left child or not
     */
    int parent_;
    /*! \brief pointer to left and right children */
    int cleft_, cright_;
    /*! \brief feature split type */
    SplitFeatureType split_type_;
    /*!
     * \brief feature index used for the split
     * highest bit indicates default direction for missing values
     */
    unsigned sindex_;
    /*! \brief storage for leaf value or decision threshold */
    Info info_;
    /*!
     * \brief operator to use for expression of form [fval] OP [threshold].
     * If the expression evaluates to true, take the left child;
     * otherwise, take the right child.
     */
    Operator cmp_;
    /*!
     * \brief list of all categories belonging to the left node.
     * Categories not in this list will belong to the right node.
     * Categories are integers ranging from 0 to (n-1), where n is the number of
     * categories in that particular feature. Let's assume n <= 64.
     */
    std::vector<uint8_t> left_categories_;
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
  /*! \brief number of nodes */
  int num_nodes;
  /*!
   * \brief get node given nid
   * \param nid node id
   * \return reference to node
   */
  inline Node& operator[](int nid) {
    return nodes[nid];
  }
  /*!
   * \brief get node given nid (const version)
   * \param nid node id
   * \return const reference to node
   */
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

  /*!
   * \brief get list of all categorical features that have appeared anywhere
   *        in tree
   */
  inline std::vector<unsigned> GetCategoricalFeatures() const {
    std::unordered_map<unsigned, bool> tmp;
    for (int nid = 0; nid < num_nodes; ++nid) {
      const Node& node = nodes[nid];
      const SplitFeatureType type = node.split_type();
      if (type != SplitFeatureType::kNone) {
        const bool flag = (type == SplitFeatureType::kCategorical);
        const unsigned split_index = node.split_index();
        if (tmp.count(split_index) == 0) {
          tmp[split_index] = flag;
        } else {
          CHECK_EQ(tmp[split_index], flag) << "Feature " << split_index
            << " cannot be simultaneously be categorical and numerical.";
        }
      }
    }
    std::vector<unsigned> result;
    for (const auto& kv : tmp) {
      if (kv.second) {
        result.push_back(kv.first);
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  }
};

struct ModelParam : public dmlc::Parameter<ModelParam> {
  DMLC_DECLARE_PARAMETER(ModelParam) {
    /* Empty for now -- can add more parameters later */
  }
};

inline void InitParamAndCheck(ModelParam* param,
                  const std::vector<std::pair<std::string, std::string>> cfg) {
  auto unknown = param->InitAllowUnknown(cfg);
  if (unknown.size() > 0) {
    std::ostringstream oss;
    for (const auto& kv : unknown) {
      oss << kv.first << ", ";
    }
    LOG(INFO) << "\033[1;31mWarning: Unknown parameters found; "
              << "they have been ignored\u001B[0m: " << oss.str();
  }
}

/*! \brief thin wrapper for tree ensemble model */
struct Model {
  /*! \brief type of multiclass classification; kNA if not applicable */
  enum class MulticlassType {
    kNA, kGradientBoosting, kRandomForest
  };
  /*! \brief member trees */
  std::vector<Tree> trees;
  /*!
   * \brief number of features used for the model.
   * It is assumed that all feature indices are between 0 and [num_feature]-1.
   */
  int num_feature;
  /*! \brief number of output groups -- for multi-class classification */
  int num_output_group;
  /*! \brief type of multiclass classification; kNA if not applicable */
  MulticlassType multiclass_type;
  /*! \brief extra parameters */
  ModelParam param;

  /*! \brief disable copy; use default move */
  Model() = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;
};

}  // namespace treelite
#endif  // TREELITE_TREE_H_
