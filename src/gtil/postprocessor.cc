/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file postprocessor.cc
 * \author Hyunsu Cho
 * \brief Functions to post-process prediction results
 */
#include "./postprocessor.h"

#include <treelite/tree.h>

#include <cmath>
#include <cstddef>
#include <unordered_map>

namespace treelite::gtil {

namespace detail::postprocessor {

template <typename InputT>
void identity(treelite::Model const&, std::int32_t, InputT*) {}

template <typename InputT>
void signed_square(treelite::Model const&, std::int32_t, InputT* elem) {
  InputT const margin = *elem;
  *elem = std::copysign(margin * margin, margin);
}

template <typename InputT>
void hinge(treelite::Model const&, std::int32_t, InputT* elem) {
  *elem = (*elem > 0 ? InputT(1) : InputT(0));
}

template <typename InputT>
void sigmoid(treelite::Model const& model, std::int32_t, InputT* elem) {
  InputT const val = *elem;
  *elem = InputT(1) / (InputT(1) + std::exp(-model.sigmoid_alpha * val));
}

template <typename InputT>
void exponential(treelite::Model const&, std::int32_t, InputT* elem) {
  *elem = std::exp(*elem);
}

template <typename InputT>
void exponential_standard_ratio(treelite::Model const& model, std::int32_t, InputT* elem) {
  *elem = std::exp2(-*elem / model.ratio_c);
}

template <typename InputT>
void logarithm_one_plus_exp(treelite::Model const&, std::int32_t, InputT* elem) {
  *elem = std::log1p(std::exp(*elem));
}

template <typename InputT>
void identity_multiclass(treelite::Model const&, std::int32_t, InputT*) {}

template <typename InputT>
void softmax(treelite::Model const&, std::int32_t num_class, InputT* row) {
  float max_margin = row[0];
  double norm_const = 0.0;
  float t;
  for (std::int32_t i = 1; i < num_class; ++i) {
    if (row[i] > max_margin) {
      max_margin = row[i];
    }
  }
  for (std::int32_t i = 0; i < num_class; ++i) {
    t = std::exp(row[i] - max_margin);
    norm_const += t;
    row[i] = t;
  }
  for (std::int32_t i = 0; i < num_class; ++i) {
    row[i] /= static_cast<float>(norm_const);
  }
}

template <typename InputT>
void multiclass_ova(treelite::Model const& model, std::int32_t num_class, InputT* row) {
  for (std::int32_t i = 0; i < num_class; ++i) {
    row[i] = InputT(1) / (InputT(1) + std::exp(-model.sigmoid_alpha * row[i]));
  }
}

}  // namespace detail::postprocessor

template <typename InputT>
PostProcessorFunc<InputT> GetPostProcessorFunc(std::string const& name) {
  if (name == "identity") {
    return detail::postprocessor::identity<InputT>;
  } else if (name == "signed_square") {
    return detail::postprocessor::signed_square<InputT>;
  } else if (name == "hinge") {
    return detail::postprocessor::hinge<InputT>;
  } else if (name == "sigmoid") {
    return detail::postprocessor::sigmoid<InputT>;
  } else if (name == "exponential") {
    return detail::postprocessor::exponential<InputT>;
  } else if (name == "exponential_standard_ratio") {
    return detail::postprocessor::exponential_standard_ratio<InputT>;
  } else if (name == "logarithm_one_plus_exp") {
    return detail::postprocessor::logarithm_one_plus_exp<InputT>;
  } else if (name == "identity_multiclass") {
    return detail::postprocessor::identity_multiclass<InputT>;
  } else if (name == "softmax") {
    return detail::postprocessor::softmax<InputT>;
  } else if (name == "multiclass_ova") {
    return detail::postprocessor::multiclass_ova<InputT>;
  } else {
    TREELITE_LOG(FATAL) << "Post-processor named '" << name << "' not found";
  }
  return nullptr;
}

template PostProcessorFunc<float> GetPostProcessorFunc(std::string const&);
template PostProcessorFunc<double> GetPostProcessorFunc(std::string const&);

}  // namespace treelite::gtil
