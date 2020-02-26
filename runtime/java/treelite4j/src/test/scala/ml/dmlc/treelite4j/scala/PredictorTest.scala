package ml.dmlc.treelite4j.scala

import ml.dmlc.treelite4j.java.PredictorTest.LoadArrayFromText
import ml.dmlc.treelite4j.java.{BatchBuilder, Entry, NativeLibLoader}
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
    predictor.numOutputGroup shouldEqual 1
    predictor.numFeature shouldEqual 127
    predictor.predTransform shouldEqual "sigmoid"
    predictor.sigmoidAlpha shouldEqual 1.0f
    predictor.globalBias shouldEqual 0.0f
  }

  test("PredictBatch") {
    val predictor = Predictor(mushroomLibLocation)
    val dmat = BatchBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation)
    val sparseBatch = BatchBuilder.CreateSparseBatch(dmat.iterator())
    val denseBatch = BatchBuilder.CreateDenseBatch(dmat.iterator())
    val retProb = LoadArrayFromText(mushroomTestDataPredProbResultLocation)
    val retMargin = LoadArrayFromText(mushroomTestDataPredMarginResultLocation)

    val sparseMargin = predictor.predictSparseBatch(sparseBatch, predMargin = true)
    val sparseProb = predictor.predictSparseBatch(sparseBatch)
    val denseMargin = predictor.predictDenseBatch(denseBatch, predMargin = true)
    val denseProb = predictor.predictDenseBatch(denseBatch)

    retProb.zip(denseProb.zip(sparseProb)).foreach { case (ret, (dense, sparse)) =>
      Seq(dense.length, sparse.length) shouldEqual Seq(1, 1)
      Seq(dense.head, sparse.head) shouldEqual Seq(ret, ret)
    }
    retMargin.zip(denseMargin.zip(sparseMargin)).foreach { case (ret, (dense, sparse)) =>
      Seq(dense.length, sparse.length) shouldEqual Seq(1, 1)
      Seq(dense.head, sparse.head) shouldEqual Seq(ret, ret)
    }
  }

  test("PredictInst") {
    val predictor = Predictor(mushroomLibLocation)
    mushroomLibPredictionTest(predictor)
  }

  private def mushroomLibPredictionTest(predictor: Predictor): Unit = {
    val instArray = Array.tabulate(predictor.numFeature)(_ => {
      val entry = new Entry()
      entry.setMissing()
      entry
    })
    val expectedResult = LoadArrayFromText(mushroomTestDataPredProbResultLocation)
    val dataPoints = BatchBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation)
    dataPoints.asScala.zipWithIndex.foreach { case (dp, row) =>
      dp.indices.zip(dp.values).foreach { case (i, v) => instArray(i).setFValue(v) }
      val result = predictor.predictInst(instArray)
      result.length shouldEqual 1
      result(0) shouldEqual expectedResult(row)
      instArray.foreach(_.setMissing())
    }
  }
}
