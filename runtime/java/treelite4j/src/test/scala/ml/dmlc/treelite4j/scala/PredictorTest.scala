package ml.dmlc.treelite4j.scala

import ml.dmlc.treelite4j.java.PredictorTest.LoadArrayFromText
import ml.dmlc.treelite4j.java.{DMatrixBuilder, Entry, NativeLibLoader}
import org.scalatest.{FunSuite, Matchers}

import scala.collection.JavaConverters._

class PredictorTest extends FunSuite with Matchers {
  private val mushroomLibLocation = NativeLibLoader
      .createTempFileFromResource("/mushroom_example/" + System.mapLibraryName("mushroom"))
  private val mushroomTestDataLocation = NativeLibLoader
      .createTempFileFromResource("/mushroom_example/agaricus.txt.test")
  private val mushroomTestDataPredProbResultLocation = NativeLibLoader
      .createTempFileFromResource("/mushroom_example/agaricus.txt.test.prob")
  private val mushroomTestDataPredMarginResultLocation = NativeLibLoader
      .createTempFileFromResource("/mushroom_example/agaricus.txt.test.margin")

  test("Basic") {
    val predictor = Predictor(mushroomLibLocation)
    predictor.numClass shouldEqual 1
    predictor.numFeature shouldEqual 127
    predictor.predTransform shouldEqual "sigmoid"
    predictor.sigmoidAlpha shouldEqual 1.0f
    predictor.ratioC shouldEqual 1.0f
    predictor.globalBias shouldEqual 0.0f
  }

  test("PredictBatch") {
    val predictor = Predictor(mushroomLibLocation)
    val dmat = DMatrixBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation)
    val sparseDMatrix = DMatrixBuilder.createSparseCSRDMatrix(dmat.iterator())
    val denseDMatrix = DMatrixBuilder.createDenseDMatrix(dmat.iterator())
    val retProb = LoadArrayFromText(mushroomTestDataPredProbResultLocation)
    val retMargin = LoadArrayFromText(mushroomTestDataPredMarginResultLocation)

    val sparseMargin = predictor.predictBatch(sparseDMatrix, predMargin = true).toFloatMatrix
    val sparseProb = predictor.predictBatch(sparseDMatrix).toFloatMatrix
    val denseMargin = predictor.predictBatch(denseDMatrix, predMargin = true).toFloatMatrix
    val denseProb = predictor.predictBatch(denseDMatrix).toFloatMatrix

    retProb.zip(denseProb.zip(sparseProb)).foreach { case (ret, (dense, sparse)) =>
      Seq(dense.length, sparse.length) shouldEqual Seq(1, 1)
      Seq(dense.head, sparse.head) shouldEqual Seq(ret, ret)
    }
    retMargin.zip(denseMargin.zip(sparseMargin)).foreach { case (ret, (dense, sparse)) =>
      Seq(dense.length, sparse.length) shouldEqual Seq(1, 1)
      Seq(dense.head, sparse.head) shouldEqual Seq(ret, ret)
    }
  }
}
