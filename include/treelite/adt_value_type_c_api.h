/*!
 * Copyright 2019 by Contributors
 * \file adt_value_type_c_api.h
 * \brief Define enum constants for the ADT (Abstract Data Type) of Treelite. ADT enables type
 *        erasure so that we can accommodate multiple data types for thresholds and leaf values
 *        without littering the whole codebase with templates.
 * Make sure that this header is fully C compatible.
 * \author Philip Cho
 */

#ifndef TREELITE_ADT_VALUE_TYPE_C_API_H
#define TREELITE_ADT_VALUE_TYPE_C_API_H

enum TreeliteValueType {
  kTreeliteInt32 = 0,
  kTreeliteFloat32 = 1,
  kTreeliteFloat64 = 2,
};

#endif  // TREELITE_ADT_VALUE_TYPE_C_API_H
