package ml.dmlc.treelite4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;
import junit.framework.TestCase;
import org.junit.Test;
import org.apache.commons.io.LineIterator;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.ArrayUtils;

/**
 * test cases for treelite predictor
 *
 * @author Philip Cho
 */
public class PredictorTest {
  private final String mushroomLibLocation;
  private final String mushroomTestDataLocation;
  private final String mushroomTestDataPredResultLocation;
  public PredictorTest() throws IOException {
    mushroomLibLocation = NativeLibLoader.createTempFileFromResource(
      "/mushroom_example/" + System.mapLibraryName("mushroom"));
    mushroomTestDataLocation = NativeLibLoader.createTempFileFromResource(
      "/mushroom_example/agaricus.txt.test");
    mushroomTestDataPredResultLocation
      = NativeLibLoader.createTempFileFromResource(
          "/mushroom_example/agaricus.txt.test.res");
  }

  @Test
  public void testPredictorBasic() throws TreeliteError {
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true, true);
    TestCase.assertEquals(1, predictor.GetNumOutputGroup());
    TestCase.assertEquals(127, predictor.GetNumFeature());
  }

  @Test
  public void testPredict() throws TreeliteError, IOException {
    Predictor predictor = new Predictor(mushroomLibLocation, 1, true, true);
    SparseBatch batch = LoadSparseBatchFromLibSVM(mushroomTestDataLocation);
    float[] expected_result
      = LoadArrayFromText(mushroomTestDataPredResultLocation);
    float[][] result = predictor.predict(batch, true, true);
    int num_row = result.length;
    for (int i = 0; i < num_row; ++i) {
      TestCase.assertEquals(1, result[i].length);
      TestCase.assertEquals(expected_result[i], result[i][0]);
    }
  }

  private float[] LoadArrayFromText(String filename) throws IOException {
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
    return ArrayUtils.toPrimitive(data.toArray(new Float[data.size()]));
  }

  private SparseBatch LoadSparseBatchFromLibSVM(String filename)
       throws TreeliteError, IOException {
    File file = new File(filename);
    LineIterator it = FileUtils.lineIterator(file, "UTF-8");
    ArrayList<Float> data = new ArrayList<Float>();
    ArrayList<Integer> col_ind = new ArrayList<Integer>();
    ArrayList<Long> row_ptr = new ArrayList<Long>();
    int num_row = 0;
    int num_col = 0;
    row_ptr.add(0L);
    try {
      while (it.hasNext()) {
        String line = it.nextLine();
        String[] tokens = line.split(" ");
        // ignore label; just read feature values

        int nnz_current_row = 0;
          // count number of nonzero feature values for current row
        for (int i = 1; i < tokens.length; ++i) {
          String[] subtokens = tokens[i].split(":");
          int fid = Integer.parseInt(subtokens[0]);
          float fval = Float.parseFloat(subtokens[1]);
          data.add(fval);
          col_ind.add(fid);
          num_col = Math.max(num_col, fid + 1);
          ++nnz_current_row;
        }
        row_ptr.add(row_ptr.get(row_ptr.size() - 1) + (long)nnz_current_row);
        ++num_row;
      }
    } finally {
      it.close();
    }
    float[] data_arr
      = ArrayUtils.toPrimitive(data.toArray(new Float[data.size()]));
    int[] col_ind_arr
      = ArrayUtils.toPrimitive(col_ind.toArray(new Integer[col_ind.size()]));
    long[] row_ptr_arr
      = ArrayUtils.toPrimitive(row_ptr.toArray(new Long[row_ptr.size()]));
    return new SparseBatch(data_arr, col_ind_arr, row_ptr_arr, num_row, num_col);
  }
}
