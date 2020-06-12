package ml.dmlc.treelite4j.java;

import junit.framework.TestCase;
import ml.dmlc.treelite4j.DataPoint;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.lang.ArrayUtils;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Test cases for Treelite Predictor
 *
 * @author Hyunsu Cho
 */
public class PredictorTest {
  private final String mushroomLibLocation;
  private final String mushroomTestDataLocation;
  private final String mushroomTestDataPredProbResultLocation;
  private final String mushroomTestDataPredMarginResultLocation;

  public PredictorTest() throws IOException {
    mushroomLibLocation = NativeLibLoader.createTempFileFromResource(
        "/mushroom_example/" + System.mapLibraryName("mushroom"));
    mushroomTestDataLocation = NativeLibLoader.createTempFileFromResource(
        "/mushroom_example/agaricus.txt.test");
    mushroomTestDataPredProbResultLocation
        = NativeLibLoader.createTempFileFromResource(
        "/mushroom_example/agaricus.txt.test.prob");
    mushroomTestDataPredMarginResultLocation
        = NativeLibLoader.createTempFileFromResource(
        "/mushroom_example/agaricus.txt.test.margin");
  }

  @Test
  public void testPredictorBasic() throws TreeliteError {
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true);
    TestCase.assertEquals(1, predictor.GetNumOutputGroup());
    TestCase.assertEquals(127, predictor.GetNumFeature());
    TestCase.assertEquals("sigmoid", predictor.GetPredTransform());
    TestCase.assertEquals(1.0f, predictor.GetSigmoidAlpha());
    TestCase.assertEquals(0.0f, predictor.GetGlobalBias());
  }

  @Test
  public void testPredict() throws TreeliteError, IOException {
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true);
    List<DataPoint> dmat
        = BatchBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation);
    SparseBatch sparse_batch = BatchBuilder.CreateSparseBatch(dmat.iterator());
    DenseBatch dense_batch = BatchBuilder.CreateDenseBatch(dmat.iterator());
    float[] expected_result
        = LoadArrayFromText(mushroomTestDataPredProbResultLocation);

    /* sparse batch */
    float[][] result = predictor.predict(sparse_batch, true, false);
    for (int i = 0; i < result.length; ++i) {
      TestCase.assertEquals(1, result[i].length);
      TestCase.assertEquals(expected_result[i], result[i][0]);
    }

    /* dense batch */
    result = predictor.predict(dense_batch, true, false);
    for (int i = 0; i < result.length; ++i) {
      TestCase.assertEquals(1, result[i].length);
      TestCase.assertEquals(expected_result[i], result[i][0]);
    }
  }

  @Test
  public void testPredictMargin() throws TreeliteError, IOException {
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true);
    List<DataPoint> dmat
        = BatchBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation);
    SparseBatch sparse_batch = BatchBuilder.CreateSparseBatch(dmat.iterator());
    DenseBatch dense_batch = BatchBuilder.CreateDenseBatch(dmat.iterator());
    float[] expected_result
        = LoadArrayFromText(mushroomTestDataPredMarginResultLocation);

    /* sparse batch */
    float[][] result = predictor.predict(sparse_batch, true, true);
    for (int i = 0; i < result.length; ++i) {
      TestCase.assertEquals(1, result[i].length);
      TestCase.assertEquals(expected_result[i], result[i][0]);
    }

    /* dense batch */
    result = predictor.predict(dense_batch, true, true);
    for (int i = 0; i < result.length; ++i) {
      TestCase.assertEquals(1, result[i].length);
      TestCase.assertEquals(expected_result[i], result[i][0]);
    }
  }

  @Test
  public void testPredictInst() throws TreeliteError, IOException {
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true);
    mushroomLibPredictionTest(predictor);
  }

  @Test
  public void testSerialization() throws TreeliteError, IOException, ClassNotFoundException {
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true);
    Predictor predictor2 = (Predictor) fromByteArray(toByteArray(predictor));
    TestCase.assertEquals(predictor.GetNumFeature(), predictor2.GetNumFeature());
    TestCase.assertEquals(predictor.GetNumOutputGroup(), predictor2.GetNumOutputGroup());
    TestCase.assertEquals(predictor.GetPredTransform(), predictor2.GetPredTransform());
    TestCase.assertEquals(predictor.GetSigmoidAlpha(), predictor2.GetSigmoidAlpha());
    TestCase.assertEquals(predictor.GetGlobalBias(), predictor2.GetGlobalBias());
    mushroomLibPredictionTest(predictor2);
  }

  private void mushroomLibPredictionTest(Predictor predictor) throws IOException, TreeliteError {
    Entry[] inst_arr = new Entry[predictor.GetNumFeature()];
    for (int i = 0; i < inst_arr.length; ++i) {
      inst_arr[i] = new Entry();
      inst_arr[i].setMissing();
    }

    float[] expected_result
        = LoadArrayFromText(mushroomTestDataPredProbResultLocation);

    List<DataPoint> dmat
        = BatchBuilder.LoadDatasetFromLibSVM(mushroomTestDataLocation);
    int row_id = 0;
    for (DataPoint inst : dmat) {
      int[] indices = inst.indices();
      float[] values = inst.values();
      for (int i = 0; i < indices.length; ++i) {
        inst_arr[indices[i]].setFValue(values[i]);
      }
      float[] result = predictor.predict(inst_arr, false);
      TestCase.assertEquals(1, result.length);
      TestCase.assertEquals(expected_result[row_id++], result[0]);
      for (int i = 0; i < inst_arr.length; ++i) {
        inst_arr[i].setMissing();
      }
    }
  }

  public static float[] LoadArrayFromText(String filename) throws IOException {
    File file = new File(filename);
    LineIterator it = FileUtils.lineIterator(file, "UTF-8");
    ArrayList<Float> data = new ArrayList<Float>();
    try {
      while (it.hasNext()) {
        float val = Float.parseFloat(it.nextLine());
        data.add(val);
      }
    } finally {
      it.close();
    }
    return ArrayUtils.toPrimitive(data.toArray(new Float[0]));
  }

  /**
   * Read the object from ByteArray.
   */
  private static Object fromByteArray(byte[] data) throws IOException,
      ClassNotFoundException {
    ObjectInputStream ois = new ObjectInputStream(
        new BufferedInputStream(new ByteArrayInputStream(data)));
    Object o = ois.readObject();
    ois.close();
    return o;
  }

  /**
   * Write the object to a ByteArray.
   */
  private static byte[] toByteArray(Serializable o) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ObjectOutputStream oos = new ObjectOutputStream(baos);
    oos.writeObject(o);
    oos.close();
    byte[] ba = baos.toByteArray();
    return ba;
  }
}
