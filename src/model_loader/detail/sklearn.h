/*!
 * Copyright (c) 2026 by Contributors
 * \file sklearn.h
 * \brief Helpers for loading scikit-learn trees
 */
#ifndef SRC_MODEL_LOADER_DETAIL_SKLEARN_H_
#define SRC_MODEL_LOADER_DETAIL_SKLEARN_H_

namespace treelite::model_loader::sklearn::detail {
inline double SplitGain(
    double const* weight, double const* impurity, int node, int left, int right) {
  // Scale the local impurity decrease by the fraction of tree weight reaching this node.
  // Sklearn impurity_improvement:
  //
  // N_t/N (I_p - N_{t,L}/N I_L - N_{t,R}/N I_R)
  return (weight[node] / weight[0])
         * (impurity[node] - (weight[left] / weight[node]) * impurity[left]
             - (weight[right] / weight[node]) * impurity[right]);
}
}  // namespace treelite::model_loader::sklearn::detail

#endif  // SRC_MODEL_LOADER_DETAIL_SKLEARN_H_
