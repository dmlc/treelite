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
[Under construction]
