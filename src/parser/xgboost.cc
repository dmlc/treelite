/*!
 * Copyright 2017 by Contributors
 * \file xgboost.cc
 * \brief Parser for xgboost model
 * \author Philip Cho
 */

#include <treelite/parser.h>
#include <memory>
#include <queue>
#include <cstring>

/* auxiliary data structures to interpret xgboost model file */
namespace {

typedef float bst_float;

/* peekable input stream implemented with a ring buffer */
class PeekableInputStream {
 public:
  const size_t MAX_PEEK_WINDOW = 1024;  // peek up to 1024 bytes

  PeekableInputStream(dmlc::Stream* fi)
   : istm_(fi), buf_(MAX_PEEK_WINDOW + 1), begin_ptr_(0), end_ptr_(0) {}

  inline size_t Read(void* ptr, size_t size) {
    const size_t bytes_buffered = BytesBuffered();
    char* cptr = static_cast<char*>(ptr);
    if (size <= bytes_buffered) {
      // all content already buffered; consume buffer
      if (begin_ptr_ + size < MAX_PEEK_WINDOW + 1) {
        std::memcpy(cptr, &buf_[begin_ptr_], size);
        begin_ptr_ += size;
      } else {
        std::memcpy(cptr, &buf_[begin_ptr_], MAX_PEEK_WINDOW + 1 - begin_ptr_);
        std::memcpy(cptr + MAX_PEEK_WINDOW + 1 - begin_ptr_, &buf_[0],
                    size + begin_ptr_ - MAX_PEEK_WINDOW - 1);
        begin_ptr_ = size + begin_ptr_ - MAX_PEEK_WINDOW - 1;
      }
      return size;
    } else {  // consume buffer entirely and read more bytes
      const size_t bytes_to_read = size - bytes_buffered;
      if (begin_ptr_ <= end_ptr_) {
        std::memcpy(cptr, &buf_[begin_ptr_], bytes_buffered);
      } else {
        std::memcpy(cptr, &buf_[begin_ptr_], MAX_PEEK_WINDOW + 1 - begin_ptr_);
        std::memcpy(cptr + MAX_PEEK_WINDOW + 1 - begin_ptr_, &buf_[0],
                    bytes_buffered + begin_ptr_ - MAX_PEEK_WINDOW - 1);
      }
      begin_ptr_ = end_ptr_;
      return bytes_buffered
             + istm_->Read(cptr + bytes_buffered, bytes_to_read);
    }
  }

  inline size_t PeekRead(void* ptr, size_t size) {
    CHECK_LE(size, MAX_PEEK_WINDOW)
      << "PeekableInputStream allows peeking up to "
      << MAX_PEEK_WINDOW << " bytes";
    char* cptr = static_cast<char*>(ptr);
    const size_t bytes_buffered = BytesBuffered();
    /* fill buffer with additional bytes, up to size */
    if (size > bytes_buffered) {
      const size_t bytes_to_read = size - bytes_buffered;
      if (end_ptr_ + bytes_to_read < MAX_PEEK_WINDOW + 1) {
        CHECK_EQ(istm_->Read(&buf_[end_ptr_], bytes_to_read), bytes_to_read)
          << "Failed to peek " << size << " bytes";
        end_ptr_ += bytes_to_read;
      } else {
        CHECK_EQ(  istm_->Read(&buf_[end_ptr_], 
                               MAX_PEEK_WINDOW + 1 - end_ptr_)
                 + istm_->Read(&buf_[0],
                               bytes_to_read + end_ptr_ - MAX_PEEK_WINDOW - 1),
                 bytes_to_read)
          << "Failed to peek " << size << " bytes";
        end_ptr_ = bytes_to_read + end_ptr_ - MAX_PEEK_WINDOW - 1;
      }
    }
    /* copy buffer into ptr without emptying buffer */
    if (begin_ptr_ <= end_ptr_) {  // usual case
      std::memcpy(cptr, &buf_[begin_ptr_], end_ptr_ - begin_ptr_);
    } else {  // context wrapped around the end
      std::memcpy(cptr, &buf_[begin_ptr_], MAX_PEEK_WINDOW + 1 - begin_ptr_);
      std::memcpy(cptr + MAX_PEEK_WINDOW + 1 - begin_ptr_, &buf_[0], end_ptr_);
    }

    return size;
  }

 private:
  dmlc::Stream* istm_;
  std::vector<char> buf_;
  size_t begin_ptr_, end_ptr_;

  inline size_t BytesBuffered() {
    if (begin_ptr_ <= end_ptr_) {  // usual case
      return end_ptr_ - begin_ptr_;
    } else {  // context wrapped around the end
      return MAX_PEEK_WINDOW + 1 + end_ptr_ - begin_ptr_;
    }
  }
};

template <typename T>
inline void CONSUME_BYTES(const T& fi, size_t size) {
  static std::vector<char> dummy(500);
  if (size > dummy.size()) dummy.resize(size);
  CHECK_EQ(fi->Read(&dummy[0], size), size) << "BoostLearner: wrong model format";
}

struct GBTreeModelParam {
  int num_trees;
  int pad0;
  int num_feature;
  int pad1;
  int64_t pad2;
  int pad3[34];
};

struct TreeParam {
  int num_roots;
  int num_nodes;
  int num_deleted;
  int pad0[2];
  int size_leaf_vector;
  int reserved[31];
};

class XGBTree {
 public:
  class Node {
   public:
    Node() : sindex_(0) {
      // assert compact alignment
      static_assert(sizeof(Node) == 4 * sizeof(int) + sizeof(Info),
                    "Node: 64 bit align");
    }
    inline int cleft() const {
      return this->cleft_;
    }
    inline int cright() const {
      return this->cright_;
    }
    inline int cdefault() const {
      return this->default_left() ? this->cleft() : this->cright();
    }
    inline unsigned split_index() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    inline bool default_left() const {
      return (sindex_ >> 31) != 0;
    }
    inline bool is_leaf() const {
      return cleft_ == -1;
    }
    inline bst_float leaf_value() const {
      return (this->info_).leaf_value;
    }
    inline bst_float split_cond() const {
      return (this->info_).split_cond;
    }
    inline int parent() const {
      return parent_ & ((1U << 31) - 1);
    }
    inline bool is_root() const {
      return parent_ == -1;
    }

   private:
    friend class XGBTree;
    union Info {
      bst_float leaf_value;
      bst_float split_cond;
    };
    int parent_;
    int cleft_, cright_;
    unsigned sindex_;
    Info info_;

    inline bool is_deleted() const {
      return sindex_ == std::numeric_limits<unsigned>::max();
    }
  };

 private:
  TreeParam param;
  std::vector<Node> nodes;

 public:
  inline Node& operator[](int nid) {
    return nodes[nid];
  }
  inline const Node& operator[](int nid) const {
    return nodes[nid];
  }
  inline void Load(PeekableInputStream* fi) {
    CHECK_EQ(fi->Read(&param, sizeof(TreeParam)), sizeof(TreeParam));
    nodes.resize(param.num_nodes);
    CHECK_NE(param.num_nodes, 0);
    CHECK_EQ(fi->Read(dmlc::BeginPtr(nodes), sizeof(Node) * nodes.size()),
             sizeof(Node) * nodes.size());
    CONSUME_BYTES(fi, (3 * sizeof(bst_float) + sizeof(int)) * param.num_nodes);
    if (param.size_leaf_vector != 0) {
      uint64_t len;
      CHECK_EQ(fi->Read(&len, sizeof(len)), sizeof(len));
      if (len > 0) {
        CONSUME_BYTES(fi, sizeof(bst_float) * len);
      }
    }
    CHECK_EQ(param.num_roots, 1)
      << "treelite does not support trees with multiple roots";
  }
};

}  // namespace anonymous

namespace treelite {
namespace parser {

DMLC_REGISTRY_FILE_TAG(xgboost);

class XGBParser : public Parser {
 public:
  XGBParser() { LOG(INFO) << "XGBParser yah"; }

  void Load(dmlc::Stream* fi) override {
    std::unique_ptr<PeekableInputStream> fp(new PeekableInputStream(fi));
    // backward compatible header check.
    std::string header;
    header.resize(4);
    if (fp->PeekRead(&header[0], 4) == 4) {
      CHECK_NE(header, "bs64")
          << "Base64 format is no longer supported in brick.";
      if (header == "binf") {
        CONSUME_BYTES(fp, 4);
      }
    }
    // read parameter
    CONSUME_BYTES(fp, sizeof(bst_float) + sizeof(unsigned) + 32 * sizeof(int));
    {
      // backward compatibility code for compatible with old model type
      // for new model, Read(&name_obj_) is suffice
      uint64_t len;
      CHECK_EQ(fp->Read(&len, sizeof(len)), sizeof(len));
      if (len >= std::numeric_limits<unsigned>::max()) {
        int gap;
        CHECK_EQ(fp->Read(&gap, sizeof(gap)), sizeof(gap))
            << "BoostLearner: wrong model format";
        len = len >> static_cast<uint64_t>(32UL);
      }
      if (len != 0) {
        name_obj_.resize(len);
        CHECK_EQ(fp->Read(&name_obj_[0], len), len)
            << "BoostLearner: wrong model format";
      }
    }

    {
      uint64_t len;
      CHECK_EQ(fp->Read(&len, sizeof(len)), sizeof(len));
      name_gbm_.resize(len);
      if (len > 0) {
        CHECK_EQ(fp->Read(&name_gbm_[0], len), len)
          << "BoostLearner: wrong model format";
      }
    }

    /* loading GBTree */
    CHECK_EQ(name_gbm_, "gbtree")
      << "Gradient booster must be gbtree type.";

    CHECK_EQ(fp->Read(&gbm_param_, sizeof(gbm_param_)), sizeof(gbm_param_))
        << "GBTree: invalid model file";
    LOG(INFO) << "gbm_param_.num_feature = " << gbm_param_.num_feature;
    xgb_trees_.clear();
    for (int i = 0; i < gbm_param_.num_trees; ++i) {
      xgb_trees_.emplace_back();
      xgb_trees_.back().Load(fp.get());
    }
  }

  std::vector<Tree> Export() const override {
    std::vector<Tree> model;
    for (const auto& xgb_tree : xgb_trees_) {
      model.emplace_back();
      Tree& tree = model.back();
      tree.Init();

      // re-map node ID's to eliminate gaps in numbering
      // deleted nodes will not be exported
      std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
      Q.push({0, 0});
      while (!Q.empty()) {
        int old_id, new_id;
        std::tie(old_id, new_id) = Q.front(); Q.pop();
        const XGBTree::Node& node = xgb_tree[old_id];
        if (node.is_leaf()) {
          tree[new_id].set_leaf(static_cast<tl_float>(node.leaf_value()));
        } else {
          tree.AddChilds(new_id);
          tree[new_id].set_split(node.split_index(),
                                 static_cast<tl_float>(node.split_cond()),
                                 node.default_left(), Tree::Operator::kLT);
          Q.push({node.cleft(), tree[new_id].cleft()});
          Q.push({node.cright(), tree[new_id].cright()});
        }
      }
    }

    return model;
  }
 private:
  std::vector<XGBTree> xgb_trees_;
  GBTreeModelParam gbm_param_;
  std::string name_gbm_;
  std::string name_obj_;
};

TREELITE_REGISTER_PARSER(XGBParser, "xgboost")
.describe("Parser for xgboost binary format")
.set_body([]() {
    return new XGBParser();
  });
}  // namespace parser
}  // namespace treelite
