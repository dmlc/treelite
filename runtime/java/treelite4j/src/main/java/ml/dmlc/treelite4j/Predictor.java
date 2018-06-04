package ml.dmlc.treelite4j;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import java.io.File;
import java.io.IOException;
import java.io.ByteArrayOutputStream;
import java.nio.file.Paths;

/**
 * Treelite predictor
 * @author Philip Cho
 */
class Predictor {
  private static final Log logger = LogFactory.getLog(Predictor.class);
  private long handle = 0;
  private int num_output_group;
  private int num_feature;

  public Predictor(
    String libpath, int nthread, boolean verbose, boolean include_master_thread)
      throws TreeliteError {
    File f = new File(libpath);
    String path = "";
    if (f.isDirectory()) {  // libpath is a diectory
      // directory is given; locate shared library inside it
      String basename = f.getName();
      boolean lib_found = false;
      for (String ext : new String[]{".so", ".dll", ".dylib"}) {
        path = Paths.get(libpath, basename + ext).toString();
        File f2 = new File(path);
        if (f2.exists()) {
          lib_found = true;
          break;
        }
      }
      if (!lib_found) {
        throw new TreeliteError(String.format(
          "Directory %s doesn't appear to have any dynamic " +
          "shared library (.so/.dll/.dylib).", libpath));
      }
    } else {  // libpath is actually the name of shared library file
      String fileext = libpath.substring(libpath.lastIndexOf('.'));
      if (fileext.equals(".dll")
          || fileext.equals(".so") || fileext.equals(".dylib")) {
        path = libpath;
      } else {
        throw new TreeliteError(String.format(
          "Specified path %s has wrong file extension (%s); the shared " +
          "library must have one of the following extensions: " +
          ".so / .dll / .dylib", libpath, fileext));
      }
    }

    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorLoad(
      path, nthread, include_master_thread, out));
    handle = out[0];

    // Save # of output groups and # of features
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryNumOutputGroup(
      handle, out));
    num_output_group = (int)out[0];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryNumFeature(
      handle, out));
    num_feature = (int)out[0];

    if (verbose) {
      logger.info(String.format(
        "Dynamic shared library %s has been successfully loaded into memory",
        path));
    }
  }

  public int GetNumOutputGroup() {
    return this.num_output_group;
  }

  public int GetNumFeature() {
    return this.num_feature;
  }

  public float[] predict(Entry[] inst, boolean pred_margin)
    throws TreeliteError, IOException {

    assert inst.length > 0;

    // query result size
    long[] out = new long[1];
    TreeliteJNI.checkCall(
      TreeliteJNI.TreelitePredictorQueryResultSizeSingleInst(this.handle, out));
    int result_size = (int)out[0];
    float[] out_result = new float[result_size];

    // serialize instance as byte array
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    for (int i = 0; i < inst.length; ++i) {
      inst[i].write(os);
    }
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorPredictInst(
      this.handle, os.toByteArray(), pred_margin, out_result, out));
    int actual_result_size = (int)out[0];
    float[] result = new float[actual_result_size];
    for (int i = 0; i < actual_result_size; ++i) {
      result[i] = out_result[i];
    }
    return result;
  }

  public float[][] predict(
    SparseBatch batch, boolean verbose, boolean pred_margin)
      throws TreeliteError {
    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryResultSize(
      this.handle, batch.getHandle(), true, out));
    int result_size = (int)out[0];
    float[] out_result = new float[result_size];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorPredictBatch(
      this.handle, batch.getHandle(), true, verbose, pred_margin,
      out_result, out));
    int actual_result_size = (int)out[0];
    return reshape(out_result, actual_result_size, this.num_output_group);
  }

  public float[][] predict(
    DenseBatch batch, boolean verbose, boolean pred_margin)
      throws TreeliteError {
    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryResultSize(
      this.handle, batch.getHandle(), false, out));
    int result_size = (int)out[0];
    float[] out_result = new float[result_size];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorPredictBatch(
      this.handle, batch.getHandle(), false, verbose, pred_margin,
      out_result, out));
    int actual_result_size = (int)out[0];
    return reshape(out_result, actual_result_size, this.num_output_group);
  }

  private float[][] reshape(float[] array, int rend, int num_col) {
    assert rend <= array.length;
    assert rend % num_col == 0;
    float[][] res;
    res = new float[rend / num_col][num_col];
    for (int i = 0; i < rend; ++i) {
      int r = i / num_col;
      int c = i % num_col;
      res[r][c] = array[i];
    }
    return res;
  }

  @Override
  protected void finalize() throws Throwable {
    super.finalize();
    dispose();
  }

  public synchronized void dispose() {
    if (handle != 0L) {
      TreeliteJNI.TreelitePredictorFree(handle);
      handle = 0;
    }
  }
}
