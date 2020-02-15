package ml.dmlc.treelite4j.scala

import java.io.IOException

import ml.dmlc.treelite4j.java.{Data, DenseBatch, SparseBatch, TreeliteError, Predictor => JPredictor}

import scala.reflect.ClassTag

/**
 * Scala wrapper of ml.dmlc.treelite4j.java.Predictor, keeping same interface as Java Predictor.
 *
 * DEVELOPER WARNING: A Java Predictor must not be shared by more than one Scala Predictor.
 * @param pred the Java Predictor object.
 */
class Predictor private[treelite4j](private[treelite4j] val pred: JPredictor)
    extends Serializable {

  @throws(classOf[TreeliteError])
  def numFeature: Int = pred.GetNumFeature()

  @throws(classOf[TreeliteError])
  def numOutputGroup: Int = pred.GetNumOutputGroup()

  @throws(classOf[TreeliteError])
  def predTransform: String = pred.GetPredTransform()

  @throws(classOf[TreeliteError])
  def sigmoidAlpha: Float = pred.GetSigmoidAlpha()

  @throws(classOf[TreeliteError])
  def globalBias: Float = pred.GetGlobalBias()

  @throws(classOf[TreeliteError])
  @throws(classOf[IOException])
  def predictInst[T <: Data : ClassTag](
      inst: Array[T],
      predMargin: Boolean = false): Array[Float] = {
    pred.predict(inst.asInstanceOf[Array[Data]], predMargin)
  }

  @throws(classOf[TreeliteError])
  def predictSparseBatch(
      batch: SparseBatch,
      predMargin: Boolean = false,
      verbose: Boolean = false): Array[Array[Float]] = {
    pred.predict(batch, predMargin, verbose)
  }

  @throws(classOf[TreeliteError])
  def predictDenseBatch(
      batch: DenseBatch,
      predMargin: Boolean = false,
      verbose: Boolean = false): Array[Array[Float]] = {
    pred.predict(batch, predMargin, verbose)
  }

  override def finalize(): Unit = {
    super.finalize()
    dispose()
  }

  def dispose(): Unit = pred.dispose()
}
