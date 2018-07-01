package ml.dmlc.treelite4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import junit.framework.TestCase;
import org.junit.Test;
import org.apache.commons.io.LineIterator;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.ArrayUtils;

/**
 * Test cases for Treelite Predictor
 *
 * @author Philip Cho
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
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true, true);
    TestCase.assertEquals(1, predictor.GetNumOutputGroup());
    TestCase.assertEquals(127, predictor.GetNumFeature());
  }

  @Test
  public void testPredict() throws TreeliteError, IOException {
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true, true);
    List<List<MatrixEntry>> dmat
      = LoadDatasetFromLibSVM(mushroomTestDataLocation);
    SparseBatch sparse_batch = CreateSparseBatch(dmat);
    DenseBatch dense_batch = CreateDenseBatch(dmat);
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
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true, true);
    List<List<MatrixEntry>> dmat
      = LoadDatasetFromLibSVM(mushroomTestDataLocation);
    SparseBatch sparse_batch = CreateSparseBatch(dmat);
    DenseBatch dense_batch = CreateDenseBatch(dmat);
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
    Predictor predictor = new Predictor(mushroomLibLocation, -1, true, true);
    Entry[] inst_arr = new Entry[predictor.GetNumFeature()];
    for (int i = 0; i < inst_arr.length; ++i) {
      inst_arr[i] = new Entry();
      inst_arr[i].setMissing();
    }

    float[] expected_result
      = LoadArrayFromText(mushroomTestDataPredProbResultLocation);

    List<List<MatrixEntry>> dmat
      = LoadDatasetFromLibSVM(mushroomTestDataLocation);
    int row_id = 0;
    for (List<MatrixEntry> inst : dmat) {
      for (MatrixEntry e : inst) {
        inst_arr[e.fid].setFValue(e.fval);
      }
      float[] result = predictor.predict(inst_arr, false);
      TestCase.assertEquals(1, result.length);
      TestCase.assertEquals(expected_result[row_id++], result[0]);
      for (int i = 0; i < inst_arr.length; ++i) {
        inst_arr[i].setMissing();
      }
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

  private class MatrixEntry {  // (feature id, feature value) pair
    int fid;
    float fval;
    public MatrixEntry(int fid, float fval) {
      this.fid = fid;
      this.fval = fval;
    }
  }

  private List<List<MatrixEntry>> LoadDatasetFromLibSVM(String filename)
       throws TreeliteError, IOException {
    File file = new File(filename);
    LineIterator it = FileUtils.lineIterator(file, "UTF-8");
    ArrayList<List<MatrixEntry>> dmat = new ArrayList<List<MatrixEntry>>();
    try {
      while (it.hasNext()) {
        String line = it.nextLine();
        String[] tokens = line.split(" ");
        ArrayList<MatrixEntry> inst = new ArrayList<MatrixEntry>();
        // ignore label; just read feature values
        for (int i = 1; i < tokens.length; ++i) {
          String[] subtokens = tokens[i].split(":");
          int fid = Integer.parseInt(subtokens[0]);
          float fval = Float.parseFloat(subtokens[1]);
          inst.add(new MatrixEntry(fid, fval));
        }
        dmat.add(inst);
      }
    } finally {
      it.close();
    }
    return dmat;
  }

  private SparseBatch CreateSparseBatch(List<List<MatrixEntry>> dmat)
       throws TreeliteError, IOException {
    ArrayList<Float> data = new ArrayList<Float>();
    ArrayList<Integer> col_ind = new ArrayList<Integer>();
    ArrayList<Long> row_ptr = new ArrayList<Long>();
    int num_row = dmat.size();
    int num_col = 0;
    row_ptr.add(0L);
    for (List<MatrixEntry> inst : dmat) {
      int nnz_current_row = 0;
        // count number of nonzero feature values for current row
      for (MatrixEntry e : inst) {
        data.add(e.fval);
        col_ind.add(e.fid);
        num_col = Math.max(num_col, e.fid + 1);
        ++nnz_current_row;
      }
      row_ptr.add(row_ptr.get(row_ptr.size() - 1) + (long)nnz_current_row);
    }
    float[] data_arr
      = ArrayUtils.toPrimitive(data.toArray(new Float[data.size()]));
    int[] col_ind_arr
      = ArrayUtils.toPrimitive(col_ind.toArray(new Integer[col_ind.size()]));
    long[] row_ptr_arr
      = ArrayUtils.toPrimitive(row_ptr.toArray(new Long[row_ptr.size()]));
    return new SparseBatch(data_arr, col_ind_arr, row_ptr_arr, num_row, num_col);
  }

  private DenseBatch CreateDenseBatch(List<List<MatrixEntry>> dmat)
       throws TreeliteError, IOException {
    int num_row = dmat.size();
    int num_col = 0;
    // compute num_col
    for (List<MatrixEntry> inst : dmat) {
      for (MatrixEntry e : inst) {
        num_col = Math.max(num_col, e.fid + 1);
      }
    }
    float[] data = new float[num_row * num_col];
    Arrays.fill(data, Float.NaN);
    // write nonzero entries
    int row_id = 0;
    for (List<MatrixEntry> inst : dmat) {
      for (MatrixEntry e : inst) {
        data[row_id * num_col + e.fid] = e.fval;
      }
      ++row_id;
    }
    return new DenseBatch(data, Float.NaN, num_row, num_col);
  }
}
