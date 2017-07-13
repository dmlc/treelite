/*!
 * Copyright by Contributors
 * \file parser.h
 * \brief Interface of parser that reads from a file stream
 *        and outputs a tree ensemble model
 * \author Philip Cho
 */
#ifndef TREELITE_PARSER_H_
#define TREELITE_PARSER_H_

#include <dmlc/registry.h>
#include <dmlc/io.h>
#include <functional>
#include <vector>
#include "./tree.h"

namespace treelite {
/*!
 * \brief interface of parser
 */
class Parser {
 public:
  /*! \brief virtual destructor */
  virtual ~Parser() {}
  /*!
   * \brief load model from stream
   * \param fi input stream.
   */
  virtual void Load(dmlc::Stream* fi) = 0;
  /*!
   * \brief export model as in-memory representation
   * \return in-memory representation of model
   */
  virtual std::vector<Tree> Export() const = 0;
  /*!
   * \brief create a parser from given name
   * \param name name of parser
   * \return The created parser
   */
  static Parser* Create(const std::string& name);
};

/*!
 * \brief Registry entry for parser
 */
struct ParserReg
    : public dmlc::FunctionRegEntryBase<ParserReg, std::function<Parser* ()> > {
};

/*!
 * \brief Macro to register parser.
 *
 * \code
 * // example of registering the xgboost parser
 * TREELITE_REGISTER_PARSER(XGBParser, "xgboost")
 * .describe("Parser for xgboost binary format")
 * .set_body([]() {
 *     return new XGBParser();
 *   });
 * \endcode
 */
#define TREELITE_REGISTER_PARSER(UniqueId, Name)                            \
  static DMLC_ATTRIBUTE_UNUSED ::treelite::ParserReg &          \
  __make_ ## ParserReg ## _ ## UniqueId ## __ =                \
      ::dmlc::Registry< ::treelite::ParserReg>::Get()->__REGISTER__(Name)

}  // namespace treelite
#endif  // TREELITE_PARSER_H_
