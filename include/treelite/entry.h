/*!
 *  Copyright (c) 2017-2020 by Contributors
 * \file entry.h
 * \author Hyunsu Cho
 * \brief Entry type for Treelite predictor
 */
#ifndef TREELITE_ENTRY_H_
#define TREELITE_ENTRY_H_

/*! \brief data layout. The value -1 signifies the missing value.
    When the "missing" field is set to -1, the "fvalue" field is set to
    NaN (Not a Number), so there is no danger for mistaking between
    missing values and non-missing values. */
union TreelitePredictorEntry {
  int missing;
  float fvalue;
  // may contain extra fields later, such as qvalue
};

#endif  // TREELITE_ENTRY_H_
