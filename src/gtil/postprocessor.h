/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file postprocessor.h
 * \author Hyunsu Cho
 * \brief Functions to post-process prediction results
 */

#ifndef SRC_GTIL_POSTPROCESSOR_H_
#define SRC_GTIL_POSTPROCESSOR_H_

#include <string>

namespace treelite {

class Model;

namespace gtil {

template <typename InputT>
using PostProcessorFunc = void (*)(treelite::Model const&, std::int32_t, InputT*);

template <typename InputT>
PostProcessorFunc<InputT> GetPostProcessorFunc(std::string const& name);

extern template PostProcessorFunc<float> GetPostProcessorFunc(std::string const& name);
extern template PostProcessorFunc<double> GetPostProcessorFunc(std::string const& name);

}  // namespace gtil
}  // namespace treelite

#endif  // SRC_GTIL_POSTPROCESSOR_H_
