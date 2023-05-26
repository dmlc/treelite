/*!
 * Copyright (c) 2023 by Contributors
 * \file serializer.h
 * \brief Implementation for model serializers
 * \author Hyunsu Cho
 */

#ifndef TREELITE_DETAIL_SERIALIZER_H_
#define TREELITE_DETAIL_SERIALIZER_H_

#include <treelite/contiguous_array.h>
#include <treelite/logging.h>
#include <treelite/pybuffer_frame.h>
#include <treelite/task_type.h>
#include <treelite/tree.h>
#include <treelite/typeinfo.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace treelite {

class Model;

}  // namespace treelite

namespace treelite::detail::serializer {

inline PyBufferFrame GetPyBufferFromArray(
    void* data, char const* format, std::size_t itemsize, std::size_t nitem) {
  return PyBufferFrame{data, const_cast<char*>(format), itemsize, nitem};
}

// Infer format string from data type
template <typename T>
inline char const* InferFormatString() {
  switch (sizeof(T)) {
  case 1:
    return (std::is_unsigned<T>::value ? "=B" : "=b");
  case 2:
    return (std::is_unsigned<T>::value ? "=H" : "=h");
  case 4:
    if (std::is_integral<T>::value) {
      return (std::is_unsigned<T>::value ? "=L" : "=l");
    } else {
      if (!std::is_floating_point<T>::value) {
        throw Error("Could not infer format string");
      }
      return "=f";
    }
  case 8:
    if (std::is_integral<T>::value) {
      return (std::is_unsigned<T>::value ? "=Q" : "=q");
    } else {
      if (!std::is_floating_point<T>::value) {
        throw Error("Could not infer format string");
      }
      return "=d";
    }
  default:
    throw Error("Unrecognized type");
  }
  return nullptr;
}

template <typename T>
inline PyBufferFrame GetPyBufferFromArray(ContiguousArray<T>* vec, char const* format) {
  return GetPyBufferFromArray(static_cast<void*>(vec->Data()), format, sizeof(T), vec->Size());
}

template <typename T>
inline PyBufferFrame GetPyBufferFromArray(ContiguousArray<T>* vec) {
  static_assert(std::is_arithmetic<T>::value,
      "Use GetPyBufferFromArray(vec, format) for composite types; specify format string manually");
  return GetPyBufferFromArray(vec, InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(void* data, char const* format, std::size_t itemsize) {
  return GetPyBufferFromArray(data, format, itemsize, 1);
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T* scalar, char const* format) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  return GetPyBufferFromScalar(static_cast<void*>(scalar), format, sizeof(T));
}

inline PyBufferFrame GetPyBufferFromScalar(TypeInfo* scalar) {
  using T = std::underlying_type<TypeInfo>::type;
  return GetPyBufferFromScalar(reinterpret_cast<T*>(scalar), InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(TaskType* scalar) {
  using T = std::underlying_type<TaskType>::type;
  return GetPyBufferFromScalar(reinterpret_cast<T*>(scalar), InferFormatString<T>());
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T* scalar) {
  static_assert(std::is_arithmetic<T>::value,
      "Use GetPyBufferFromScalar(scalar, format) for composite types; "
      "specify format string manually");
  return GetPyBufferFromScalar(scalar, InferFormatString<T>());
}

template <typename T>
inline void InitArrayFromPyBuffer(ContiguousArray<T>* vec, PyBufferFrame frame) {
  if (sizeof(T) != frame.itemsize) {
    throw Error("Incorrect itemsize");
  }
  vec->UseForeignBuffer(frame.buf, frame.nitem);
}

inline void InitScalarFromPyBuffer(TypeInfo* scalar, PyBufferFrame buffer) {
  using T = std::underlying_type<TypeInfo>::type;
  if (sizeof(T) != buffer.itemsize) {
    throw Error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw Error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = static_cast<TypeInfo>(*t);
}

inline void InitScalarFromPyBuffer(TaskType* scalar, PyBufferFrame buffer) {
  using T = std::underlying_type<TaskType>::type;
  if (sizeof(T) != buffer.itemsize) {
    throw Error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw Error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = static_cast<TaskType>(*t);
}

template <typename T>
inline void InitScalarFromPyBuffer(T* scalar, PyBufferFrame buffer) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  if (sizeof(T) != buffer.itemsize) {
    throw Error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw Error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = *t;
}

template <typename T>
inline void ReadScalarFromStream(T* scalar, std::istream& is) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  is.read(reinterpret_cast<char*>(scalar), sizeof(T));
}

template <typename T>
inline void WriteScalarToStream(T* scalar, std::ostream& os) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  os.write(reinterpret_cast<char const*>(scalar), sizeof(T));
}

template <typename T>
inline void ReadArrayFromStream(ContiguousArray<T>* vec, std::istream& is) {
  std::uint64_t nelem;
  is.read(reinterpret_cast<char*>(&nelem), sizeof(nelem));
  vec->Clear();
  vec->Resize(nelem);
  if (nelem == 0) {
    return;  // handle empty arrays
  }
  is.read(reinterpret_cast<char*>(vec->Data()), sizeof(T) * nelem);
}

template <typename T>
inline void WriteArrayToStream(ContiguousArray<T>* vec, std::ostream& os) {
  static_assert(sizeof(std::uint64_t) >= sizeof(std::size_t), "size_t too large");
  auto const nelem = static_cast<std::uint64_t>(vec->Size());
  os.write(reinterpret_cast<char const*>(&nelem), sizeof(nelem));
  if (nelem == 0) {
    return;  // handle empty arrays
  }
  os.write(reinterpret_cast<char const*>(vec->Data()), sizeof(T) * vec->Size());
}

inline void SkipOptFieldInStream(std::istream& is) {
  std::uint16_t elem_size;
  std::uint64_t nelem;
  ReadScalarFromStream(&elem_size, is);
  ReadScalarFromStream(&nelem, is);

  const std::uint64_t nbytes = elem_size * nelem;
  TREELITE_CHECK_LE(nbytes, std::numeric_limits<std::streamoff>::max());  // NOLINT
  is.seekg(static_cast<std::streamoff>(nbytes), std::ios::cur);
}

class StreamSerializerMixIn {
 public:
  explicit StreamSerializerMixIn(std::ostream& os) : os_(os) {}

  template <typename T>
  void SerializePrimitiveField(T* field) {
    WriteScalarToStream(field, os_);
  }

  template <typename T>
  void SerializeCompositeField(T* field, char const*) {
    WriteScalarToStream(field, os_);
  }

  template <typename T>
  void SerializePrimitiveArray(T* field) {
    WriteArrayToStream(field, os_);
  }

  template <typename T>
  void SerializeCompositeArray(T* field, char const*) {
    WriteArrayToStream(field, os_);
  }

 private:
  std::ostream& os_;
};

class StreamDeserializerMixIn {
 public:
  explicit StreamDeserializerMixIn(std::istream& is) : is_(is) {}

  template <typename T>
  void DeserializePrimitiveField(T* field) {
    ReadScalarFromStream(field, is_);
  }

  template <typename T>
  void DeserializeCompositeField(T* field) {
    ReadScalarFromStream(field, is_);
  }

  template <typename T>
  void DeserializePrimitiveArray(T* field) {
    ReadArrayFromStream(field, is_);
  }

  template <typename T>
  void DeserializeCompositeArray(T* field) {
    ReadArrayFromStream(field, is_);
  }

  void SkipOptionalField() {
    SkipOptFieldInStream(is_);
  }

 private:
  std::istream& is_;
};

class PyBufferSerializerMixIn {
 public:
  PyBufferSerializerMixIn() = default;

  template <typename T>
  void SerializePrimitiveField(T* field) {
    frames_.push_back(GetPyBufferFromScalar(field));
  }

  template <typename T>
  void SerializeCompositeField(T* field, char const* format) {
    frames_.push_back(GetPyBufferFromScalar(field, format));
  }

  template <typename T>
  void SerializePrimitiveArray(T* field) {
    frames_.push_back(GetPyBufferFromArray(field));
  }

  template <typename T>
  void SerializeCompositeArray(T* field, char const* format) {
    frames_.push_back(GetPyBufferFromArray(field, format));
  }

  std::vector<PyBufferFrame> GetFrames() {
    return frames_;
  }

 private:
  std::vector<PyBufferFrame> frames_;
};

class PyBufferDeserializerMixIn {
 public:
  explicit PyBufferDeserializerMixIn(std::vector<PyBufferFrame> const& frames)
      : it_(frames.cbegin()) {}

  template <typename T>
  void DeserializePrimitiveField(T* field) {
    InitScalarFromPyBuffer(field, *it_++);
  }

  template <typename T>
  void DeserializeCompositeField(T* field) {
    InitScalarFromPyBuffer(field, *it_++);
  }

  template <typename T>
  void DeserializePrimitiveArray(T* field) {
    InitArrayFromPyBuffer(field, *it_++);
  }

  template <typename T>
  void DeserializeCompositeArray(T* field) {
    InitArrayFromPyBuffer(field, *it_++);
  }

  void SkipOptionalField() {
    ++it_;
  }

 private:
  std::vector<PyBufferFrame>::const_iterator it_;
};

template <typename MixIn>
class Serializer {
 public:
  explicit Serializer(MixIn& mixin) : mixin_(mixin) {}

  void SerializeHeader(Model& model) {
    // Header 1
    model.major_ver_ = TREELITE_VER_MAJOR;
    model.minor_ver_ = TREELITE_VER_MINOR;
    model.patch_ver_ = TREELITE_VER_PATCH;
    mixin_.SerializePrimitiveField(&model.major_ver_);
    mixin_.SerializePrimitiveField(&model.minor_ver_);
    mixin_.SerializePrimitiveField(&model.patch_ver_);
    mixin_.SerializePrimitiveField(&model.threshold_type_);
    mixin_.SerializePrimitiveField(&model.leaf_output_type_);

    // Number of trees
    mixin_.SerializePrimitiveField(&model.num_tree_);

    // Header 2
    mixin_.SerializePrimitiveField(&model.num_feature);
    mixin_.SerializePrimitiveField(&model.task_type);
    mixin_.SerializePrimitiveField(&model.average_tree_output);
    mixin_.SerializeCompositeField(&model.task_param, "T{=B=?xx=I=I}");
    mixin_.SerializeCompositeField(
        &model.param, "T{" _TREELITE_STR(TREELITE_MAX_PRED_TRANSFORM_LENGTH) "s=f=f=f}");

    /* Extension Slot 1: Per-model optional fields -- to be added later */
    model.num_opt_field_per_model_ = 0;
    mixin_.SerializePrimitiveField(&model.num_opt_field_per_model_);
  }

  void SerializeTrees(Model& model) {
    std::visit(
        [&](auto&& concrete_model) {
          TREELITE_CHECK_EQ(concrete_model.trees.size(), model.num_tree_)
              << "Incorrect number of trees in the model";
          for (auto& tree : concrete_model.trees) {
            SerializeTree(tree);
          }
        },
        model.variant_);
  }

  template <typename ThresholdType, typename LeafOutputType>
  void SerializeTree(Tree<ThresholdType, LeafOutputType>& tree) {
    mixin_.SerializePrimitiveField(&tree.num_nodes);
    mixin_.SerializePrimitiveField(&tree.has_categorical_split_);
    mixin_.SerializeCompositeArray(&tree.nodes_, tree.GetFormatStringForNode());
    mixin_.SerializePrimitiveArray(&tree.leaf_vector_);
    mixin_.SerializePrimitiveArray(&tree.leaf_vector_begin_);
    mixin_.SerializePrimitiveArray(&tree.leaf_vector_end_);
    mixin_.SerializePrimitiveArray(&tree.matching_categories_);
    mixin_.SerializePrimitiveArray(&tree.matching_categories_offset_);

    /* Extension slot 2: Per-tree optional fields -- to be added later */
    tree.num_opt_field_per_tree_ = 0;
    mixin_.SerializePrimitiveField(&tree.num_opt_field_per_tree_);

    /* Extension slot 3: Per-node optional fields -- to be added later */
    tree.num_opt_field_per_node_ = 0;
    mixin_.SerializePrimitiveField(&tree.num_opt_field_per_node_);
  }

 private:
  MixIn& mixin_;
};

template <typename MixIn>
class Deserializer {
 public:
  explicit Deserializer(MixIn& mixin) : mixin_(mixin) {}

  std::unique_ptr<Model> DeserializeHeaderAndCreateModel() {
    // Header 1
    std::int32_t major_ver, minor_ver, patch_ver;
    mixin_.DeserializePrimitiveField(&major_ver);
    mixin_.DeserializePrimitiveField(&minor_ver);
    mixin_.DeserializePrimitiveField(&patch_ver);
    if (major_ver != TREELITE_VER_MAJOR && !(major_ver == 3 && minor_ver == 9)) {
      TREELITE_LOG(FATAL) << "Cannot load model from a different major Treelite version or "
                          << "a version before 3.9.0." << std::endl
                          << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
                          << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
                          << "The model checkpoint was generated from Treelite version "
                          << major_ver << "." << minor_ver << "." << patch_ver;
    } else if (major_ver == TREELITE_VER_MAJOR && minor_ver > TREELITE_VER_MINOR) {
      TREELITE_LOG(WARNING)
          << "The model you are loading originated from a newer Treelite version; some "
          << "functionalities may be unavailable." << std::endl
          << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
          << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
          << "The model checkpoint was generated from Treelite version " << major_ver << "."
          << minor_ver << "." << patch_ver;
    }
    TypeInfo threshold_type, leaf_output_type;
    mixin_.DeserializePrimitiveField(&threshold_type);
    mixin_.DeserializePrimitiveField(&leaf_output_type);

    std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
    model->major_ver_ = major_ver;
    model->minor_ver_ = minor_ver;
    model->patch_ver_ = patch_ver;

    // Number of trees
    mixin_.DeserializePrimitiveField(&model->num_tree_);

    // Header 2
    mixin_.DeserializePrimitiveField(&model->num_feature);
    mixin_.DeserializePrimitiveField(&model->task_type);
    mixin_.DeserializePrimitiveField(&model->average_tree_output);
    mixin_.DeserializeCompositeField(&model->task_param);
    mixin_.DeserializeCompositeField(&model->param);

    /* Extension Slot 1: Per-model optional fields -- to be added later */
    bool const use_opt_field = (major_ver >= 3);
    if (use_opt_field) {
      mixin_.DeserializePrimitiveField(&model->num_opt_field_per_model_);
      // Ignore extra fields; the input is likely from a later version of Treelite
      for (std::int32_t i = 0; i < model->num_opt_field_per_model_; ++i) {
        mixin_.SkipOptionalField();
      }
    } else {
      TREELITE_LOG(FATAL) << "Only Treelite format version 3.x or later is supported.";
    }

    return model;
  }

  void DeserializeTrees(Model& model) {
    std::visit(
        [&](auto&& concrete_model) {
          concrete_model.trees.clear();
          for (std::uint64_t i = 0; i < model.num_tree_; ++i) {
            concrete_model.trees.emplace_back();
            DeserializeTree(concrete_model.trees.back());
          }
        },
        model.variant_);
  }

  template <typename ThresholdType, typename LeafOutputType>
  void DeserializeTree(Tree<ThresholdType, LeafOutputType>& tree) {
    mixin_.DeserializePrimitiveField(&tree.num_nodes);
    mixin_.DeserializePrimitiveField(&tree.has_categorical_split_);
    mixin_.DeserializeCompositeArray(&tree.nodes_);
    TREELITE_CHECK_EQ(static_cast<std::size_t>(tree.num_nodes), tree.nodes_.Size())
        << "Could not load the correct number of nodes";
    mixin_.DeserializePrimitiveArray(&tree.leaf_vector_);
    mixin_.DeserializePrimitiveArray(&tree.leaf_vector_begin_);
    mixin_.DeserializePrimitiveArray(&tree.leaf_vector_end_);
    mixin_.DeserializePrimitiveArray(&tree.matching_categories_);
    mixin_.DeserializePrimitiveArray(&tree.matching_categories_offset_);

    /* Extension slot 2: Per-tree optional fields -- to be added later */
    mixin_.DeserializePrimitiveField(&tree.num_opt_field_per_tree_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < tree.num_opt_field_per_tree_; ++i) {
      mixin_.SkipOptionalField();
    }

    /* Extension slot 3: Per-node optional fields -- to be added later */
    mixin_.DeserializePrimitiveField(&tree.num_opt_field_per_node_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (int32_t i = 0; i < tree.num_opt_field_per_node_; ++i) {
      mixin_.SkipOptionalField();
    }
  }

 private:
  MixIn& mixin_;
};

}  // namespace treelite::detail::serializer

#endif  // TREELITE_DETAIL_SERIALIZER_H_
