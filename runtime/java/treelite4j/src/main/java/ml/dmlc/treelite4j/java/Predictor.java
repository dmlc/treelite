package ml.dmlc.treelite4j.java;

import org.apache.commons.io.FileUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Treelite Predictor
 * @author Philip Cho
 */
public class Predictor implements Serializable {
  private static final Log logger = LogFactory.getLog(Predictor.class);
  private transient long handle = 0;
  private transient int num_output_group;
  private transient int num_feature;
  private transient String pred_transform;
  private transient float sigmoid_alpha;
  private transient float global_bias;
  private transient int num_thread;
  private transient boolean verbose;
  private transient String libpath;
  private transient String libext;

  /**
   * Create a Predictor by loading a shared library (dll/so/dylib).
   * The library is expected to contain the compiled code for making prediction
   * for a particular tree ensemble model. The predictor also spawns a
   * fixed-size pool of worker threads, who will wait for prediction tasks.
   * (Note that the worker threads will go to sleep when no prediction task
   * is available, to free up CPU cycles for other processes.)
   *
   * @param libpath Path to the shared library
   * @param nthread Number of workers threads to spawn. Set to -1 to use default,
   *                i.e., to launch as many threads as CPU cores available on
   *                the system. You are not allowed to launch more threads than
   *                CPU cores. Setting ``nthread=1`` indicates that the main
   *                thread should be exclusively used.
   * @param verbose Whether to print extra diagnostic messages
   * @return Created Predictor
   * @throws TreeliteError
   */
  public Predictor(
          String libpath, int nthread, boolean verbose) throws TreeliteError {
    this.num_thread = nthread;
    this.verbose = verbose;
    initNativeLibrary(libpath);
  }

  private void initNativeLibrary(String libpath) throws TreeliteError {
    File f = new File(libpath);
    if (f.isDirectory()) {  // libpath is a diectory
      // directory is given; locate shared library inside it
      String basename = f.getName();
      boolean lib_found = false;
      for (String ext : new String[]{".so", ".dll", ".dylib"}) {
        this.libpath = Paths.get(libpath, basename + ext).toString();
        File f2 = new File(this.libpath);
        if (f2.exists()) {
          lib_found = true;
          this.libext = ext;
          break;
        }
      }
      if (!lib_found) {
        throw new TreeliteError(String.format(
                "Directory %s doesn't appear to have any dynamic " +
                        "shared library (.so/.dll/.dylib).", libpath));
      }
    } else {  // libpath is actually the name of shared library file
      this.libext = libpath.substring(libpath.lastIndexOf('.'));
      if (this.libext.equals(".dll")
          || this.libext.equals(".so") || this.libext.equals(".dylib")) {
        this.libpath = libpath;
      } else {
        throw new TreeliteError(String.format(
            "Specified path %s has wrong file extension (%s); the shared " +
                "library must have one of the following extensions: " +
                ".so / .dll / .dylib", libpath, this.libext));
      }
    }

    long[] long_out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorLoad(
            this.libpath, this.num_thread, long_out));
    handle = long_out[0];

    // Fetch meta information from model
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryNumOutputGroup(
            handle, long_out));
    num_output_group = (int) long_out[0];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryNumFeature(
            handle, long_out));
    num_feature = (int) long_out[0];
    String[] str_out = new String[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryPredTransform(
            handle, str_out));
    pred_transform = str_out[0];
    float[] fp_out = new float[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQuerySigmoidAlpha(
            handle, fp_out));
    sigmoid_alpha = fp_out[0];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryGlobalBias(
            handle, fp_out));
    global_bias = fp_out[0];

    if (this.verbose) {
      logger.info(String.format(
              "Dynamic shared library %s has been successfully loaded into memory",
              this.libpath));
    }
  }

  /**
   * Get the number of output groups for the compiled model. This number is
   * 1 for tasks other than multi-class classification. For multi-class
   * classification task, the number is equal to the number of classes.
   *
   * @return Number of output groups
   */
  public int GetNumOutputGroup() {
    return this.num_output_group;
  }

  /**
   * Get the number of features used by the compiled model. Call this method
   * to allocate array for storing data entries of a single instance.
   *
   * @return Number of features
   */
  public int GetNumFeature() {
    return this.num_feature;
  }

  /**
   * Get name of post prediction transformation used to train the loaded model.
   *
   * @return Name of (post-)prediction transformation
   */
  public String GetPredTransform() {
    return this.pred_transform;
  }

  /**
   * Get alpha value in sigmoid transformation used to train the loaded model.
   *
   * @return Alpha value of sigmoid transformation
   */
  public float GetSigmoidAlpha() {
    return this.sigmoid_alpha;
  }

  /**
   * Get global bias which adjusting predicted margin scores.
   *
   * @return Value of global bias
   */
  public float GetGlobalBias() {
    return this.global_bias;
  }

  /**
   * Perform single-instance prediction. Prediction is run by the calling thread.
   *
   * @param inst        array of data entires comprising the instance
   * @param pred_margin whether to predict a probability or a raw margin score
   * @return Resulting predictions, of dimension ``[num_output_group]``
   */
  public float[] predict(Data[] inst, boolean pred_margin)
          throws TreeliteError, IOException {

    assert inst.length > 0;

    // query result size
    long[] out = new long[1];
    TreeliteJNI.checkCall(
            TreeliteJNI.TreelitePredictorQueryResultSizeSingleInst(this.handle, out));
    int result_size = (int) out[0];
    float[] out_result = new float[result_size];

    // serialize instance as byte array
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    for (int i = 0; i < inst.length; ++i) {
      inst[i].write(os);
    }
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorPredictInst(
            this.handle, os.toByteArray(), pred_margin, out_result, out));
    int actual_result_size = (int) out[0];
    float[] result = new float[actual_result_size];
    for (int i = 0; i < actual_result_size; ++i) {
      result[i] = out_result[i];
    }
    return result;
  }

  /**
   * Perform batch prediction with a 2D sparse data matrix. Worker threads
   * will internally divide up work for batch prediction. **Note that this
   * function will be blocked by mutex when worker_thread > 1.** In order to use
   * multiple threads to process multiple prediction requests simultaneously,
   * use :java:meth:`Predictor.predict(Data[], boolean)` instead or keep worker_thread = 1.
   *
   * @param batch       a :java:ref:`SparseBatch`, representing a slice of a 2D
   *                    sparse matrix
   * @param verbose     whether to print extra diagnostic messages
   * @param pred_margin whether to predict probabilities or raw margin scores
   * @return Resulting predictions, of dimension ``[num_row]*[num_output_group]``
   */
  public float[][] predict(
          SparseBatch batch, boolean verbose, boolean pred_margin)
          throws TreeliteError {
    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryResultSize(
            this.handle, batch.getHandle(), true, out));
    int result_size = (int) out[0];
    float[] out_result = new float[result_size];
    if (num_thread == 1) {
      TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorPredictBatch(
              this.handle, batch.getHandle(), true, verbose, pred_margin,
              out_result, out));
    } else {
      synchronized (this) {
        TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorPredictBatch(
                this.handle, batch.getHandle(), true, verbose, pred_margin,
                out_result, out));
      }
    }
    int actual_result_size = (int) out[0];
    return reshape(out_result, actual_result_size, this.num_output_group);
  }

  /**
   * Perform batch prediction with a 2D dense data matrix. Worker threads
   * will internally divide up work for batch prediction. **Note that this
   * function will be blocked by mutex when worker_thread > 1.** In order to use
   * multiple threads to process multiple prediction requests simultaneously,
   * use :java:meth:`Predictor.predict(Data[], boolean)` instead or keep worker_thread = 1.
   *
   * @param batch       a :java:ref:`DenseBatch`, representing a slice of a 2D dense
   *                    matrix
   * @param verbose     whether to print extra diagnostic messages
   * @param pred_margin whether to predict probabilities or raw margin scores
   * @return Resulting predictions, of dimension ``[num_row]*[num_output_group]``
   */
  public float[][] predict(
          DenseBatch batch, boolean verbose, boolean pred_margin)
          throws TreeliteError {
    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorQueryResultSize(
            this.handle, batch.getHandle(), false, out));
    int result_size = (int) out[0];
    float[] out_result = new float[result_size];
    if (num_thread == 1) {
      TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorPredictBatch(
              this.handle, batch.getHandle(), false, verbose, pred_margin,
              out_result, out));
    } else {
      synchronized (this) {
        TreeliteJNI.checkCall(TreeliteJNI.TreelitePredictorPredictBatch(
                this.handle, batch.getHandle(), false, verbose, pred_margin,
                out_result, out));
      }
    }
    int actual_result_size = (int) out[0];
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

  /**
   * Destructor, to be called when the object is garbage collected
   */
  public synchronized void dispose() {
    if (handle != 0L) {
      TreeliteJNI.TreelitePredictorFree(handle);
      handle = 0;
    }
  }

  private void readObject(java.io.ObjectInputStream in)
          throws IOException, ClassNotFoundException {
    this.num_thread = in.readInt();
    this.verbose = in.readBoolean();
    byte[] libext = new byte[in.readShort()];
    in.read(libext);
    File libpath = File.createTempFile("TreeLite_", new String(libext));
    byte[] lib_data = new byte[in.readInt()];
    // use readFully here, because only 1024 bytes can be fetched by ObjectInputStream.read
    in.readFully(lib_data);
    FileUtils.writeByteArrayToFile(libpath, lib_data);
    try {
      initNativeLibrary(libpath.getAbsolutePath());
    } catch (TreeliteError ex) {
      ex.printStackTrace();
      logger.error("Error while loading TreeLite dynamic shared library!");
    } finally {
      libpath.delete();
    }
  }

  private void writeObject(java.io.ObjectOutputStream out) throws IOException {
    out.writeInt(this.num_thread);
    out.writeBoolean(this.verbose);
    byte[] libext = this.libext.getBytes();
    out.writeShort(libext.length);
    out.write(libext);
    byte[] lib_data = Files.readAllBytes(Paths.get(libpath));
    out.writeInt(lib_data.length);
    out.write(lib_data);
  }
}
