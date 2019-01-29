Deploying models with Treelite4J
================================

Treelite4J is the Java runtime for Treelite. This tutorial will show how to use Treelite4J to deploy decision tree models to Java applications.

Load compiled model
-------------------
Locate the compiled model (dll/so/dylib) in the local filesystem. We load the compiled model by creating a Predictor object:

.. code-block:: java

  import ml.dmlc.treelite4j.Predictor;

  Predictor predictor = new Predictor("path/to/compiled_model.so", -1, true, true);

The second argument is set to -1, to utilize all CPU cores available. See `here <https://treelite.readthedocs.io/en/latest/javadoc/ml/dmlc/treelite4j/Predictor.html#ml.dmlc.treelite4j.Predictor.Predictor(String,%20int,%20boolean,%20boolean)>`_ for the meaning of third and fourth arguments.

Query the model
---------------
Once the compiled model is loaded, we can query it:

.. code-block:: java

  // Get the input dimension, i.e. the number of feature values in the input vector
  int num_feature = predictor.GetNumFeature();

  // Get the size of output per input
  // This number is 1 for tasks other than multi-class classification.
  // For multi-class classification task, the number is equal to the number of classes.
  int num_output_group = predictor.GetNumOutputGroup();

Predict with a single input (instance)
--------------------------------------
For predicting with a single input, we create an array of Entry objects, set their values,
and invoke the prediction function.

.. code-block:: java

  import ml.dmlc.treelite4j.Entry;

  // Create an array of feature values for the input
  int num_feature = predictor.GetNumFeature();
  Entry[] inst = new Entry[num_feature];

  // Initialize all feature values as missing
  for (int i = 0; i < num_feature; ++i) {
    inst[i] = new Entry();
    inst[i].setMissing();
  }

  // Set feature values that are not missing
  // In this example, we set feature 1, 3, and 7
  inst[1].setFValue(-0.5);
  inst[3].setFValue(3.2);
  inst[7].setFValue(-1.7);

  // Now run prediction
  // (Put false in the second argument to get probability outputs)
  float[] result = predictor.predict(inst, false);
  // The result is either class probabilities (for multi-class classification)
  // or a single number (for all other tasks, such as regression)

Predict with a batch of inputs
------------------------------
For predicting with a batch of inputs, we create a list of DataPoint objects. Each DataPoint object consists of feature values and corresponding feature indices.

Let us look at an example. Consider the following 4-by-6 data matrix

.. math::

  \left[
    \begin{array}{cccccc}
      10 & 20 & \cdot & \cdot & \cdot & \cdot\\
      \cdot & 30 & \cdot & 40 & \cdot & \cdot\\
      \cdot & \cdot & 50 & 60 & 70 & \cdot\\
      \cdot & \cdot & \cdot & \cdot & \cdot & 80
    \end{array}
  \right]

where the dot (.) indicates the missing value. The matrix consists of 4 data points (instances), each with 6 feature values.
Since not all feature values are present, we need to store feature indices as well as feature values:

.. code-block:: java

  import ml.dmlc.treelite4j.DataPoint;

  // Create a list consisting of 4 data points
  List<DataPoint> dmat = new ArrayList<DataPoint>() {
    {
      //                feature indices     feature values
      add(new DataPoint(new int[]{0, 1},    new float[]{10f, 20f}));
      add(new DataPoint(new int[]{1, 3},    new float[]{30f, 40f}));
      add(new DataPoint(new int[]{2, 3, 4}, new float[]{50f, 60f, 70f}));
      add(new DataPoint(new int[]{5},       new float[]{80f}));
    }
  };

Once the list is created, we then convert it into a SparseBatch object. We use SparseBatch rather than DenseBatch because significant portion of the data matrix
consists of missing values.

.. code-block:: java

  import ml.dmlc.treelite4j.BatchBuilder;

  // Convert data point list into SparseBatch object
  SparseBatch batch = BatchBuilder.CreateSparseBatch(dmat);

Now invoke the batch prediction function using the SparseBatch object:

.. code-block:: java

  // verbose=true, pred_margin=false
  float[][] result = predictor.predict(batch, true, false);

The returned array is a two-dimensional array where the array ``result[i]`` represents the prediction for the ``i``-th data point. For most applications, each ``result[i]`` has length 1. Multi-class classification task is specical, in that for that task ``result[i]`` contains class probabilities, so the array is as long as the number of target classes.
