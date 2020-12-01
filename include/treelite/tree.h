/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file tree.h
 * \brief model structure for tree ensemble
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_H_
#define TREELITE_TREE_H_

#include <treelite/base.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>
#include <limits>
#include <cstring>
#include <cstdio>

#define __TREELITE_STR(x) #x
#define _TREELITE_STR(x) __TREELITE_STR(x)

#define TREELITE_MAX_PRED_TRANSFORM_LENGTH 256

/* Foward declarations */
namespace dmlc {

class Stream;
float stof(const std::string& value, size_t* pos);

}  // namespace dmlc

namespace treelite {

struct PyBufferFrame {
  void* buf;
  char* format;
  size_t itemsize;
  size_t nitem;
};

template <typename T>
class ContiguousArray {
 public:
  ContiguousArray();
  ~ContiguousArray();
  // NOTE: use Clone to make deep copy; copy constructors disabled
  ContiguousArray(const ContiguousArray&) = delete;
  ContiguousArray& operator=(const ContiguousArray&) = delete;
  ContiguousArray(ContiguousArray&& other) noexcept;
  ContiguousArray& operator=(ContiguousArray&& other) noexcept;
  inline ContiguousArray Clone() const;
  inline void UseForeignBuffer(void* prealloc_buf, size_t size);
  inline T* Data();
  inline const T* Data() const;
  inline T* End();
  inline const T* End() const;
  inline T& Back();
  inline const T& Back() const;
  inline size_t Size() const;
  inline void Reserve(size_t newsize);
  inline void Resize(size_t newsize);
  inline void Resize(size_t newsize, T t);
  inline void Clear();
  inline void PushBack(T t);
  inline void Extend(const std::vector<T>& other);
  inline T& operator[](size_t idx);
  inline const T& operator[](size_t idx) const;
  static_assert(std::is_pod<T>::value, "T must be POD");

 private:
  T* buffer_;
  size_t size_;
  size_t capacity_;
  bool owned_buffer_;
};

/*! \brief in-memory representation of a decision tree */
template <typename ThresholdType, typename LeafOutputType>
class Tree {
 public:
  /*! \brief tree node */
  struct Node {
    /*! \brief Initialization method.
     * Use this in lieu of constructor (POD types cannot have a non-trivial constructor) */
    inline void Init();
    /*! \brief store either leaf value or decision threshold */
    union Info {
      LeafOutputType leaf_value;  // for leaf nodes
      ThresholdType threshold;   // for non-leaf nodes
    };
    /*! \brief pointer to left and right children */
    int32_t cleft_, cright_;
    /*!
     * \brief feature index used for the split
     * highest bit indicates default direction for missing values
     */
    uint32_t sindex_;
    /*! \brief storage for leaf value or decision threshold */
    Info info_;
    /*!
     * \brief number of data points whose traversal paths include this node.
     *        LightGBM models natively store this statistics.
     */
    uint64_t data_count_;
    /*!
     * \brief sum of hessian values for all data points whose traversal paths
     *        include this node. This value is generally correlated positively
     *        with the data count. XGBoost models natively store this
     *        statistics.
     */
    double sum_hess_;
    /*!
     * \brief change in loss that is attributed to a particular split
     */
    double gain_;
    /*! \brief feature split type */
    SplitFeatureType split_type_;
    /*!
     * \brief operator to use for expression of form [fval] OP [threshold].
     * If the expression evaluates to true, take the left child;
     * otherwise, take the right child.
     */
    Operator cmp_;
    /* \brief Whether to convert missing value to zero.
     * When this flag is set, it overrides the behavior of default_left().
     */
    bool missing_value_to_zero_;
    /*! \brief whether data_count_ field is present */
    bool data_count_present_;
    /*! \brief whether sum_hess_ field is present */
    bool sum_hess_present_;
    /*! \brief whether gain_present_ field is present */
    bool gain_present_;
  };

  static_assert(std::is_pod<Node>::value, "Node must be a POD type");
  static_assert(std::is_same<ThresholdType, float>::value
                || std::is_same<ThresholdType, double>::value,
                "ThresholdType must be either float32 or float64");
  static_assert(std::is_same<LeafOutputType, uint32_t>::value
                || std::is_same<LeafOutputType, float>::value
                || std::is_same<LeafOutputType, double>::value,
                "LeafOutputType must be one of uint32_t, float32 or float64");
  static_assert(std::is_same<ThresholdType, LeafOutputType>::value
                || std::is_same<LeafOutputType, uint32_t>::value,
                "Unsupported combination of ThresholdType and LeafOutputType");
  static_assert((std::is_same<ThresholdType, float>::value && sizeof(Node) == 48)
                || (std::is_same<ThresholdType, double>::value && sizeof(Node) == 56),
                "Node size incorrect");

  Tree() = default;
  ~Tree() = default;
  Tree(const Tree&) = delete;
  Tree& operator=(const Tree&) = delete;
  Tree(Tree&&) noexcept = default;
  Tree& operator=(Tree&&) noexcept = default;

  inline const char* GetFormatStringForNode();
  inline void GetPyBuffer(std::vector<PyBufferFrame>* dest);
  inline void InitFromPyBuffer(std::vector<PyBufferFrame>::iterator begin,
                               std::vector<PyBufferFrame>::iterator end);

 private:
  // vector of nodes
  ContiguousArray<Node> nodes_;
  ContiguousArray<LeafOutputType> leaf_vector_;
  ContiguousArray<size_t> leaf_vector_offset_;
  ContiguousArray<uint32_t> left_categories_;
  ContiguousArray<size_t> left_categories_offset_;

  // allocate a new node
  inline int AllocNode();

 public:
  /*! \brief number of nodes */
  int num_nodes;
  /*! \brief initialize the model with a single root node */
  inline void Init();
  /*!
   * \brief add child nodes to node
   * \param nid node id to add children to
   */
  inline void AddChilds(int nid);

  /*!
   * \brief get list of all categorical features that have appeared anywhere in tree
   * \return list of all categorical features used
   */
  inline std::vector<unsigned> GetCategoricalFeatures() const;

  /** Getters **/
  /*!
   * \brief index of the node's left child
   * \param nid ID of node being queried
   */
  inline int LeftChild(int nid) const {
    return nodes_[nid].cleft_;
  }
  /*!
   * \brief index of the node's right child
   * \param nid ID of node being queried
   */
  inline int RightChild(int nid) const {
    return nodes_[nid].cright_;
  }
  /*!
   * \brief index of the node's "default" child, used when feature is missing
   * \param nid ID of node being queried
   */
  inline int DefaultChild(int nid) const {
    return DefaultLeft(nid) ? LeftChild(nid) : RightChild(nid);
  }
  /*!
   * \brief feature index of the node's split condition
   * \param nid ID of node being queried
   */
  inline uint32_t SplitIndex(int nid) const {
    return (nodes_[nid].sindex_ & ((1U << 31U) - 1U));
  }
  /*!
   * \brief whether to use the left child node, when the feature in the split condition is missing
   * \param nid ID of node being queried
   */
  inline bool DefaultLeft(int nid) const {
    return (nodes_[nid].sindex_ >> 31U) != 0;
  }
  /*!
   * \brief whether the node is leaf node
   * \param nid ID of node being queried
   */
  inline bool IsLeaf(int nid) const {
    return nodes_[nid].cleft_ == -1;
  }
  /*!
   * \brief get leaf value of the leaf node
   * \param nid ID of node being queried
   */
  inline LeafOutputType LeafValue(int nid) const {
    return (nodes_[nid].info_).leaf_value;
  }
  /*!
   * \brief get leaf vector of the leaf node; useful for multi-class random forest classifier
   * \param nid ID of node being queried
   */
  inline std::vector<LeafOutputType> LeafVector(int nid) const {
    if (nid > leaf_vector_offset_.Size()) {
      throw std::runtime_error("nid too large");
    }
    return std::vector<LeafOutputType>(&leaf_vector_[leaf_vector_offset_[nid]],
                                       &leaf_vector_[leaf_vector_offset_[nid + 1]]);
  }
  /*!
   * \brief tests whether the leaf node has a non-empty leaf vector
   * \param nid ID of node being queried
   */
  inline bool HasLeafVector(int nid) const {
    if (nid > leaf_vector_offset_.Size()) {
      throw std::runtime_error("nid too large");
    }
    return leaf_vector_offset_[nid] != leaf_vector_offset_[nid + 1];
  }
  /*!
   * \brief get threshold of the node
   * \param nid ID of node being queried
   */
  inline ThresholdType Threshold(int nid) const {
    return (nodes_[nid].info_).threshold;
  }
  /*!
   * \brief get comparison operator
   * \param nid ID of node being queried
   */
  inline Operator ComparisonOp(int nid) const {
    return nodes_[nid].cmp_;
  }
  /*!
   * \brief Get list of all categories belonging to the left child node. Categories not in this
   *        list will belong to the right child node. Categories are integers ranging from 0 to
   *        (n-1), where n is the number of categories in that particular feature. This list is
   *        assumed to be in ascending order.
   * \param nid ID of node being queried
   */
  inline std::vector<uint32_t> LeftCategories(int nid) const {
    if (nid > left_categories_offset_.Size()) {
      throw std::runtime_error("nid too large");
    }
    return std::vector<uint32_t>(&left_categories_[left_categories_offset_[nid]],
                                 &left_categories_[left_categories_offset_[nid + 1]]);
  }
  /*!
   * \brief get feature split type
   * \param nid ID of node being queried
   */
  inline SplitFeatureType SplitType(int nid) const {
    return nodes_[nid].split_type_;
  }
  /*!
   * \brief test whether this node has data count
   * \param nid ID of node being queried
   */
  inline bool HasDataCount(int nid) const {
    return nodes_[nid].data_count_present_;
  }
  /*!
   * \brief get data count
   * \param nid ID of node being queried
   */
  inline uint64_t DataCount(int nid) const {
    return nodes_[nid].data_count_;
  }

  /*!
   * \brief test whether this node has hessian sum
   * \param nid ID of node being queried
   */
  inline bool HasSumHess(int nid) const {
    return nodes_[nid].sum_hess_present_;
  }
  /*!
   * \brief get hessian sum
   * \param nid ID of node being queried
   */
  inline double SumHess(int nid) const {
    return nodes_[nid].sum_hess_;
  }
  /*!
   * \brief test whether this node has gain value
   * \param nid ID of node being queried
   */
  inline bool HasGain(int nid) const {
    return nodes_[nid].gain_present_;
  }
  /*!
   * \brief get gain value
   * \param nid ID of node being queried
   */
  inline double Gain(int nid) const {
    return nodes_[nid].gain_;
  }
  /*!
   * \brief test whether missing values should be converted into zero
   * \param nid ID of node being queried
   */
  inline bool MissingValueToZero(int nid) const {
    return nodes_[nid].missing_value_to_zero_;
  }

  /** Setters **/
  /*!
   * \brief create a numerical split
   * \param nid ID of node being updated
   * \param split_index feature index to split
   * \param threshold threshold value
   * \param default_left the default direction when feature is unknown
   * \param missing_value_to_zero whether missing values should be converted into zero
   * \param cmp comparison operator to compare between feature value and
   *            threshold
   */
  inline void SetNumericalSplit(int nid, unsigned split_index, ThresholdType threshold,
                                bool default_left, bool missing_value_to_zero, Operator cmp);
  /*!
   * \brief create a categorical split
   * \param nid ID of node being updated
   * \param split_index feature index to split
   * \param threshold threshold value
   * \param default_left the default direction when feature is unknown
   * \param missing_value_to_zero whether missing values should be converted into zero
   * \param cmp comparison operator to compare between feature value and
   *            threshold
   */
  inline void SetCategoricalSplit(int nid, unsigned split_index, bool default_left,
                                  bool missing_value_to_zero,
                                  const std::vector<uint32_t>& left_categories);
  /*!
   * \brief set the leaf value of the node
   * \param nid ID of node being updated
   * \param value leaf value
   */
  inline void SetLeaf(int nid, LeafOutputType value);
  /*!
   * \brief set the leaf vector of the node; useful for multi-class random forest classifier
   * \param nid ID of node being updated
   * \param leaf_vector leaf vector
   */
  inline void SetLeafVector(int nid, const std::vector<LeafOutputType>& leaf_vector);
  /*!
   * \brief set the hessian sum of the node
   * \param nid ID of node being updated
   * \param sum_hess hessian sum
   */
  inline void SetSumHess(int nid, double sum_hess) {
    Node& node = nodes_[nid];
    node.sum_hess_ = sum_hess;
    node.sum_hess_present_ = true;
  }
  /*!
   * \brief set the data count of the node
   * \param nid ID of node being updated
   * \param data_count data count
   */
  inline void SetDataCount(int nid, uint64_t data_count) {
    Node& node = nodes_[nid];
    node.data_count_ = data_count;
    node.data_count_present_ = true;
  }
  /*!
   * \brief set the gain value of the node
   * \param nid ID of node being updated
   * \param gain gain value
   */
  inline void SetGain(int nid, double gain) {
    Node& node = nodes_[nid];
    node.gain_ = gain;
    node.gain_present_ = true;
  }

  void ReferenceSerialize(dmlc::Stream* fo) const;
};

struct ModelParam {
  /*!
  * \defgroup model_param
  * Extra parameters for tree ensemble models
  * \{
  */
  /*!
   * \brief name of prediction transform function
   *
   * This parameter specifies how to transform raw margin values into
   * final predictions. By default, this is set to `'identity'`, which
   * means no transformation.
   *
   * For the **multi-class classification task**, `pred_transfrom` must be one
   * of the following values:
   * \snippet src/compiler/pred_transform.cc pred_transform_multiclass_db
   *
   * For **all other tasks** (e.g. regression, binary classification, ranking
   * etc.), `pred_transfrom` must be one of the following values:
   * \snippet src/compiler/pred_transform.cc pred_transform_db
   *
   */
  char pred_transform[TREELITE_MAX_PRED_TRANSFORM_LENGTH] = {0};
  /*!
   * \brief scaling parameter for sigmoid function
   * `sigmoid(x) = 1 / (1 + exp(-alpha * x))`
   *
   * This parameter is used only when `pred_transform` is set to `'sigmoid'`.
   * It must be strictly positive; if unspecified, it is set to 1.0.
   */
  float sigmoid_alpha;
  /*!
   * \brief global bias of the model
   *
   * Predicted margin scores of all instances will be adjusted by the global
   * bias. If unspecified, the bias is set to zero.
   */
  float global_bias;
  /*! \} */

  ModelParam() : sigmoid_alpha(1.0f), global_bias(0.0f) {
    std::memset(pred_transform, 0, TREELITE_MAX_PRED_TRANSFORM_LENGTH * sizeof(char));
    std::strncpy(pred_transform, "identity", sizeof(pred_transform));
  }
  ~ModelParam() = default;
  ModelParam(const ModelParam&) = default;
  ModelParam& operator=(const ModelParam&) = default;
  ModelParam(ModelParam&&) = default;
  ModelParam& operator=(ModelParam&&) = default;

  template<typename Container>
  inline std::vector<std::pair<std::string, std::string>>
  InitAllowUnknown(const Container &kwargs);
  inline std::map<std::string, std::string> __DICT__() const;
};

static_assert(std::is_standard_layout<ModelParam>::value,
              "ModelParam must be in the standard layout");

inline void InitParamAndCheck(ModelParam* param,
                              const std::vector<std::pair<std::string, std::string>>& cfg);

/*! \brief thin wrapper for tree ensemble model */
class Model {
 public:
  /*! \brief disable copy; use default move */
  Model() = default;
  virtual ~Model() = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;

  template <typename ThresholdType, typename LeafOutputType>
  inline static std::unique_ptr<Model> Create();
  inline static std::unique_ptr<Model> Create(TypeInfo threshold_type, TypeInfo leaf_output_type);
  inline TypeInfo GetThresholdType() const {
    return threshold_type_;
  }
  inline TypeInfo GetLeafOutputType() const {
    return leaf_output_type_;
  }
  template <typename Func>
  inline auto Dispatch(Func func);
  template <typename Func>
  inline auto Dispatch(Func func) const;

  virtual size_t GetNumTree() const = 0;
  virtual void SetTreeLimit(size_t limit) = 0;
  virtual void ReferenceSerialize(dmlc::Stream* fo) const = 0;

  inline std::vector<PyBufferFrame> GetPyBuffer();
  inline static std::unique_ptr<Model> CreateFromPyBuffer(std::vector<PyBufferFrame> frames);

  /*!
   * \brief number of features used for the model.
   * It is assumed that all feature indices are between 0 and [num_feature]-1.
   */
  int num_feature;
  /*! \brief number of output groups -- for multi-class classification
   *  Set to 1 for everything else */
  int num_output_group;
  /*! \brief flag for random forest;
   *  True for random forests and False for gradient boosted trees */
  bool random_forest_flag;
  /*! \brief extra parameters */
  ModelParam param;

 private:
  TypeInfo threshold_type_;
  TypeInfo leaf_output_type_;
  // Internal functions for serialization
  virtual void GetPyBuffer(std::vector<PyBufferFrame>* dest) = 0;
  virtual void InitFromPyBuffer(std::vector<PyBufferFrame>::iterator begin,
                                std::vector<PyBufferFrame>::iterator end) = 0;
};

template <typename ThresholdType, typename LeafOutputType>
class ModelImpl : public Model {
 public:
  /*! \brief member trees */
  std::vector<Tree<ThresholdType, LeafOutputType>> trees;

  /*! \brief disable copy; use default move */
  ModelImpl() = default;
  ~ModelImpl() override = default;
  ModelImpl(const ModelImpl&) = delete;
  ModelImpl& operator=(const ModelImpl&) = delete;
  ModelImpl(ModelImpl&&) noexcept = default;
  ModelImpl& operator=(ModelImpl&&) noexcept = default;

  void ReferenceSerialize(dmlc::Stream* fo) const override;
  inline size_t GetNumTree() const override {
    return trees.size();
  }
  void SetTreeLimit(size_t limit) override {
    return trees.resize(limit);
  }

  inline void GetPyBuffer(std::vector<PyBufferFrame>* dest) override;
  inline void InitFromPyBuffer(std::vector<PyBufferFrame>::iterator begin,
                               std::vector<PyBufferFrame>::iterator end) override;
};

}  // namespace treelite

#include "tree_impl.h"

#endif  // TREELITE_TREE_H_
