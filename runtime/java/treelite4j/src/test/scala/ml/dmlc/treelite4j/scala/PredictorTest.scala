package ml.dmlc.treelite4j.scala

import scala.collection.JavaConverters._
import ml.dmlc.treelite4j.java.PredictorTest.LoadArrayFromText
import ml.dmlc.treelite4j.java.{BatchBuilder, Entry, NativeLibLoader, Predictor => JPredictor}
import org.scalatest.{FunSuite, Matchers}

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
    val predictor = new Predictor(new JPredictor(mushroomLibLocation, -1, true))
    predictor.numOutputGroup shouldEqual 1
    predictor.numFeature shouldEqual 127
    predictor.predTransform shouldEqual "sigmoid"
    predictor.sigmoidAlpha shouldEqual 1.0f
    predictor.globalBias shouldEqual 0.0f
  }

  test("PredictBatch") {
    val predictor = new Predictor(new JPredictor(mushroomLibLocation, -1, true))
    val dmat = BatchBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation)
    val sparseBatch = BatchBuilder.CreateSparseBatch(dmat)
    val denseBatch = BatchBuilder.CreateDenseBatch(dmat)
    val expectedResult = LoadArrayFromText(mushroomTestDataPredProbResultLocation)

    /* sparse batch */
    var result = predictor.predictSparseBatch(sparseBatch, predMargin = true)
    result.zip(expectedResult).foreach { case (actual, expect) =>
      actual.length shouldEqual 1
      actual.head shouldEqual expect
    }

    /* dense batch */
    result = predictor.predictDenseBatch(denseBatch, predMargin = true)
    result.zip(expectedResult).foreach { case (actual, expect) =>
      actual.length shouldEqual 1
      actual.head shouldEqual expect
    }
  }

  test("PredictInst") {
    val predictor = new Predictor(new JPredictor(mushroomLibLocation, -1, true))
    mushroomLibPredictionTest(predictor)
  }

  private def mushroomLibPredictionTest(predictor: Predictor): Unit = {
    val instArray = Array.tabulate(predictor.numFeature)(_ => {
      val entry = new Entry()
      entry.setMissing()
      entry
    })
    val expectedResult = LoadArrayFromText(mushroomTestDataPredProbResultLocation)
    val dmat = BatchBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation)
    dmat.asScala.zipWithIndex.foreach { case (dp, row) =>
      dp.indices.zip(dp.values).foreach { case (i, v) => instArray(i).setFValue(v) }
      val result = predictor.predictInst(instArray)
      result.length shouldEqual 1
      result(0) shouldEqual expectedResult(row)
      instArray.foreach(_.setMissing())
    }
  }
}
