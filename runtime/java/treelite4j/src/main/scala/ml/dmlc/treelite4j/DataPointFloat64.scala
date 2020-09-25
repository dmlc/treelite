package ml.dmlc.treelite4j

/**
 * A data point (instance) with float64 values
 *
 * @param indices Feature indices of this point or `null` if the data is dense
 * @param values Feature values of this point
 */
case class DataPointFloat64(
    indices: Array[Int],
    values: Array[Double]) extends Serializable {
  require(indices == null || indices.length == values.length,
    "indices and values must have the same number of elements")
}
