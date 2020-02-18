package ml.dmlc.treelite4j.scala

import ml.dmlc.treelite4j.java.PredictorTest.LoadArrayFromText
import ml.dmlc.treelite4j.java.{BatchBuilder, NativeLibLoader}
import ml.dmlc.treelite4j.scala.spark.TreeLiteModel
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterEach, FunSuite, Matchers}

import scala.collection.JavaConverters._
import scala.math.min


class TreeLiteModelTest extends FunSuite with Matchers with BeforeAndAfterEach { self: FunSuite =>
  private val mushroomLibLocation = NativeLibLoader
      .createTempFileFromResource("/mushroom_example/" + System.mapLibraryName("mushroom"))
  private val mushroomTestDataLocation = NativeLibLoader
      .createTempFileFromResource("/mushroom_example/agaricus.txt.test")
  private val mushroomTestDataPredProbResultLocation = NativeLibLoader
      .createTempFileFromResource("/mushroom_example/agaricus.txt.test.prob")
  private val mushroomTestDataPredMarginResultLocation = NativeLibLoader
      .createTempFileFromResource("/mushroom_example/agaricus.txt.test.margin")

  private val numWorkers: Int = min(Runtime.getRuntime.availableProcessors(), 4)

  @transient private var currentSession: SparkSession = _

  def ss: SparkSession = getOrCreateSession
  implicit def sc: SparkContext = ss.sparkContext

  private def sparkSessionBuilder: SparkSession.Builder = SparkSession.builder()
      .master(s"local[${numWorkers}]")
      .appName("TreeLiteSuite")
      .config("spark.ui.enabled", false)
      .config("spark.driver.memory", "512m")
      .config("spark.task.cpus", 1)

  override def beforeEach(): Unit = getOrCreateSession

  override def afterEach() {
    synchronized {
      if (currentSession != null) {
        currentSession.stop()
        currentSession = null
      }
    }
  }

  private def getOrCreateSession = synchronized {
    if (currentSession == null) {
      currentSession = sparkSessionBuilder.getOrCreate()
      currentSession.sparkContext.setLogLevel("ERROR")
    }
    currentSession
  }

  private def buildDataFrame(numPartitions: Int = numWorkers): DataFrame = {
    val probResult = LoadArrayFromText(mushroomTestDataPredProbResultLocation)
    val marginResult = LoadArrayFromText(mushroomTestDataPredMarginResultLocation)
    val dataPoint = BatchBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation).asScala
    val localData = dataPoint.zip(probResult.zip(marginResult)).map {
      case (dp, (prob, margin)) =>
        val feature = Vectors.sparse(dp.indices.max + 1, dp.indices, dp.values.map(_.toDouble))
        (feature, prob, margin)
    }
    ss.createDataFrame(sc.parallelize(localData, numPartitions))
        .toDF("features", "prob", "margin")
  }

  test("TreeLiteModel basic tests") {
    val model = TreeLiteModel(mushroomLibLocation)
    model.getBatchSize shouldEqual 4096
    model.setBatchSize(1024)
    model.getBatchSize shouldEqual 1024
    model.getPredictMargin shouldEqual false
    model.setPredictMargin(true)
    model.getPredictMargin shouldEqual true
    model.getVerbose shouldEqual false
    model.setVerbose(true)
    model.getVerbose shouldEqual true
    val predictor = model.nativePredictor
    predictor.numOutputGroup shouldEqual 1
    predictor.numFeature shouldEqual 127
    predictor.predTransform shouldEqual "sigmoid"
    predictor.sigmoidAlpha shouldEqual 1.0f
    predictor.globalBias shouldEqual 0.0f
  }

  test("TreeLiteModel test transforming") {
    val df = buildDataFrame()
    val model = TreeLiteModel(mushroomLibLocation)
    val retDF = model.transform(df)
    retDF.schema.fields.map(_.name) shouldEqual Array("features", "prob", "margin", "prediction")
    retDF.collect().foreach { row =>
      row.getAs[Float]("prob") shouldEqual row.getAs[Seq[Float]]("prediction").head
    }
  }

  test("TreeLiteModel test transforming with margin") {
    val df = buildDataFrame()
    val model = TreeLiteModel(mushroomLibLocation)
    model.setPredictMargin(true)
    val retDF = model.transform(df)
    retDF.collect().foreach { row =>
      row.getAs[Float]("margin") shouldEqual row.getAs[Seq[Float]]("prediction").head
    }
  }
}
