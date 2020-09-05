package ml.dmlc.treelite4j.scala.spark

import ml.dmlc.treelite4j.DataPoint
import ml.dmlc.treelite4j.java.DMatrixBuilder
import ml.dmlc.treelite4j.scala.Predictor
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.JavaConverters._

class TreeLiteModel private[spark](
    override val uid: String,
    private val model: Predictor
) extends PredictionModel[Vector, TreeLiteModel] {

  final val batchSize: Param[Int] = new Param[Int](
    this, "batchSize", "batch size of each iteration")

  setDefault(batchSize, 4096)

  final def getBatchSize: Int = $(batchSize)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  final val predictMargin: Param[Boolean] = new Param[Boolean](
    this, "predictMargin", "whether to predict probabilities or raw margin scores")

  setDefault(predictMargin, false)

  final def getPredictMargin: Boolean = $(predictMargin)

  def setPredictMargin(value: Boolean): this.type = set(predictMargin, value)

  final val verbose: Param[Boolean] = new Param[Boolean](
    this, "verbose", "whether to print extra diagnostic messages")

  setDefault(verbose, false)

  final def getVerbose: Boolean = $(verbose)

  def setVerbose(value: Boolean): this.type = set(verbose, value)

  /**
   * Get the local predictor, from whom you can get meta information about the model.
   *
   * @return ml.dmlc.treelite4j.scala.Predictor
   */
  def nativePredictor: Predictor = model

  override protected def predict(features: Vector): Double = {
    throw new UnsupportedOperationException(
      "TreeLiteModel don't support single instance prediction!")
  }

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    // broadcast Predictor
    val broadcastModel = dataset.sparkSession.sparkContext.broadcast(model)
    // make prediction through mini-batch style
    val resultRDD = dataset.asInstanceOf[Dataset[Row]].rdd.mapPartitions { rowIterator =>
      rowIterator.grouped($(batchSize)).flatMap { batchRow =>
        val dataPoints = batchRow.iterator.map { row =>
          row.getAs[Vector]($(featuresCol)) match {
            case v: SparseVector => DataPoint(v.indices, v.values.map(_.toFloat))
            case v: DenseVector => DataPoint(null, v.values.map(_.toFloat))
          }
        }
        val result = batchRow.head.getAs[Vector]($(featuresCol)) match {
          case _: SparseVector =>
            val batch = DMatrixBuilder.createSparseCSRDMatrix(dataPoints.asJava)
            val ret = broadcastModel.value.predictBatch(batch, $(predictMargin), $(verbose))
            batch.dispose()
            ret.toFloatMatrix.map(Row.apply(_))
          case _: DenseVector =>
            val batch = DMatrixBuilder.createDenseDMatrix(dataPoints.asJava)
            val ret = broadcastModel.value.predictBatch(batch, $(predictMargin), $(verbose))
            batch.dispose()
            ret.toFloatMatrix.map(Row.apply(_))
        }
        batchRow.zip(result).map { case (origin, ret) =>
          Row.merge(origin, ret)
        }
      }
    }
    // append result columns to schema
    val schema = StructType(dataset.schema.fields ++ Seq(
      StructField($(predictionCol), ArrayType(FloatType, containsNull = false), nullable = false)))

    dataset.sparkSession.createDataFrame(resultRDD, schema)
  }

  override def copy(extra: ParamMap): TreeLiteModel = {
    val newModel = copyValues(new TreeLiteModel(uid, model), extra)
    newModel.setParent(parent)
  }

}

object TreeLiteModel {
  /**
   * @param libPath Path to the shared library
   */
  def apply(libPath: String): TreeLiteModel = {
    // Fix numThread on 1 to get rid of mutex lock on predictBatch method.
    // Let Spark take over the work to orchestrate cores(threads).
    val pred = Predictor.apply(libPath, numThread = 1)
    new TreeLiteModel(Identifiable.randomUID("treelite"), pred)
  }
}
