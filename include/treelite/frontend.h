/*!
 * Copyright 2017 by Contributors
 * \file frontend.h
 * \brief Collection of front-end methods to load or construct ensemble model
 * \author Philip Cho
 */
#ifndef TREELITE_FRONTEND_H_
#define TREELITE_FRONTEND_H_

namespace treelite {

struct Model;  // forward declaration

namespace frontend {

Model LoadLightGBMModel(const char* filename);
Model LoadXGBoostModel(const char* filename);

}  // namespace frontend
}  // namespace treelite
#endif  // TREELITE_FRONTEND_H_
