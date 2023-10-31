Specifying models using model builder
=====================================

Treelite supports loading models from major tree libraries, such as XGBoost and
scikit-learn. However, you may want to use models trained by other tree
libraries that are not directly supported by Treelite. The model builder is
useful in this use case.

.. contents:: Contents
  :local:

What is the model builder?
--------------------------
The :py:class:`~treelite.model_builder.ModelBuilder` class is a tool used to
specify decision tree ensembles programmatically.

Example: Regressor
------------------
Consider the following regression model, consisting of two trees:

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph graph1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label="Feature 0 < 5.0 ?", shape=box];
      1 [label="Feature 2 < -3.0 ?", shape=box];
      2 [label="+0.6"];
      3 [label="-0.4"];
      4 [label="+1.2"];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="No"];
      1 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
      1 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
    }
  """
  Source(source, format='svg').render('../_static/builder_toy1', view=False)

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph graph1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label="Feature 1 < 2.5 ?", shape=box];
      1 [label="+1.6"];
      2 [label="Feature 2 < -1.2 ?", shape=box];
      3 [label="+0.1"];
      4 [label="-0.3"];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
      2 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      2 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="No"];
    }
  """
  Source(source, format='svg').render('../_static/builder_toy2', view=False)


.. image:: ../_static/builder_toy1.svg

.. image:: ../_static/builder_toy2.svg

.. note:: Provision for missing data: default directions

  Decision trees in Treelite accomodate `missing data
  <https://en.wikipedia.org/wiki/Missing_data>`_ by indicating the
  **default direction** for every test node. In the diagram above, the
  default direction is indicated by label "Missing." For instance, the root node
  of the first tree shown above will send to the left all data points that lack
  values for feature 0.

  For now, let's assume that we've somehow found
  optimal choices of default directions at training time. For detailed
  instructions for actually deciding default directions, see Section 3.4
  of `the XGBoost paper <https://arxiv.org/pdf/1603.02754v3.pdf>`_.

.. role:: red

Let us construct this ensemble using the model builder. First step is to
assign **unique integer key** to each node. In the following diagram,
integer keys are indicated in :red:`red`. Note that integer keys need to be
unique only within the same tree.

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph graph1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label=<<FONT COLOR="red">0:</FONT> Feature 0 &lt; 5.0 ?>, shape=box];
      1 [label=<<FONT COLOR="red">1:</FONT> Feature 2 &lt; -3.0 ?>, shape=box];
      2 [label=<<FONT COLOR="red">2:</FONT> +0.6>];
      3 [label=<<FONT COLOR="red">3:</FONT> -0.4>];
      4 [label=<<FONT COLOR="red">4:</FONT> +1.2>];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="No"];
      1 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
      1 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
    }
  """
  Source(source, format='svg').render('../_static/builder_toy1_1', view=False)

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph graph1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label=<<FONT COLOR="red">0:</FONT> Feature 1 &lt; 2.5 ?>, shape=box];
      1 [label=<<FONT COLOR="red">1:</FONT> +1.6>];
      2 [label=<<FONT COLOR="red">2:</FONT> Feature 2 &lt; -1.2 ?>, shape=box];
      3 [label=<<FONT COLOR="red">3:</FONT> +0.1>];
      4 [label=<<FONT COLOR="red">4:</FONT> -0.3>];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
      2 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      2 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="No"];
    }
  """
  Source(source, format='svg').render('../_static/builder_toy2_1', view=False)

.. image:: ../_static/builder_toy1_1.svg

.. image:: ../_static/builder_toy2_1.svg

Next, we create a model builder object by calling the constructor for
:py:class:`~treelite.model_builder.ModelBuilder` with some model metadata.

.. code-block:: python

  import treelite
  from treelite.model_builder import (
    Metadata,
    ModelBuilder,
    PostProcessorFunc,
    TreeAnnotation,
  )
  builder = ModelBuilder(
    threshold_type="float32",
    leaf_output_type="float32",
    metadata=Metadata(
      num_feature=3,
      task_type="kRegressor",  # Regression model
      average_tree_output=False,
      num_target=1,
      num_class=[1],   # Set num_class=1 for regression model
      leaf_vector_shape=(1, 1),  # Each tree outputs a scalar
    ),
    # Every tree generates output for target 0, class 0
    tree_annotation=TreeAnnotation(num_tree=2, target_id=[0, 0], class_id=[0, 0]),
    # The link function for the output is the identity function
    postprocessor=PostProcessorFunc(name="identity"),
    # Add this value for all outputs. Also known as the intercept.
    base_scores=[0.0],
  )

The model generates output for a single output target, so we set ``num_target=1``.
Also, the model produces a continuous output, so we set ``num_class=[1]``. ``num_class`` is an array
because each output target has a unique number of classes. We will later look at an example where a
model produces multiple output targets.

For ``tree_annotation`` field, specify the number of trees you will build via ``num_tree`` argument.
Set ``target_id=[0] * num_tree`` and ``class_id=[0] * num_tree``, since each tree produces output for Target 0, Class 0.
Later, we will look an example where the tree model produces outputs for multiple targets and multiple classes.

With the builder object, we are now ready to construct the trees.

.. code-block:: python

  # Tree 0
  builder.start_tree()
  # Tree 0, Node 0
  builder.start_node(0)
  builder.numerical_test(
    feature_id=0,
    threshold=5.0,
    default_left=True,
    opname="<",
    left_child_key=1,
    right_child_key=2,
  )
  builder.end_node()
  # Tree 0, Node 1
  builder.start_node(1)
  builder.numerical_test(
    feature_id=2,
    threshold=-3.0,
    default_left=False,
    opname="<",
    left_child_key=3,
    right_child_key=4,
  )
  builder.end_node()
  # Tree 0, Node 2
  builder.start_node(2)
  builder.leaf(0.6)
  builder.end_node()
  # Tree 0, Node 3
  builder.start_node(3)
  builder.leaf(-0.4)
  builder.end_node()
  # Tree 0, Node 4
  builder.start_node(4)
  builder.leaf(1.2)
  builder.end_node()
  builder.end_tree()

  # Tree 1
  builder.start_tree()
  # Tree 1, Node 0
  builder.start_node(0)
  builder.numerical_test(
    feature_id=1,
    threshold=2.5,
    default_left=False,
    opname="<",
    left_child_key=1,
    right_child_key=2,
  )
  builder.end_node()
  # Tree 1, Node 1
  builder.start_node(1)
  builder.leaf(1.6)
  builder.end_node()
  # Tree 1, Node 2
  builder.start_node(2)
  builder.numerical_test(
    feature_id=2,
    threshold=-1.2,
    default_left=True,
    opname="<",
    left_child_key=3,
    right_child_key=4,
  )
  builder.end_node()
  # Tree 1, Node 3
  builder.start_node(3)
  builder.leaf(0.1)
  builder.end_node()
  # Tree 1, Node 4
  builder.start_node(4)
  builder.leaf(-0.3)
  builder.end_node()
  builder.end_tree()

It is important to declare the start and end of each tree and node by calling ``start_*`` and ``end_*`` methods.
Failure to do so will generate an error.

.. note:: The first node is assumed to be the root node

  You may specify the nodes in a tree in an arbitrary order. There is
  one requirement however: the first node to be specified is always
  assumed to be the root node. In the example above, node 0 is the
  root node because it is specified the first.

We are now done building the member trees. The last step is to call
:py:meth:`~treelite.model_builder.ModelBuilder.commit` to finalize the ensemble into
a :py:class:`~treelite.Model` object:

.. code-block:: python

  # Finalize and obtain Model object
  model = builder.commit()

Let's inspect the content of the model by looking at its JSON dump:

.. code-block:: python

  print(model.dump_as_json())

which produces

.. code-block:: json

  {
      "num_feature": 3,
      "task_type": "kRegressor",
      "average_tree_output": false,
      "num_target": 1,
      "num_class": [1],
      "leaf_vector_shape": [1, 1],
      "target_id": [0, 0],
      "class_id": [0, 0],
      "postprocessor": "identity",
      "sigmoid_alpha": 1.0,
      "ratio_c": 1.0,
      "base_scores": [0.0],
      "attributes": "{}",
      "trees": [{
              "num_nodes": 5,
              "has_categorical_split": false,
              "nodes": [{
                      "node_id": 0,
                      "split_feature_id": 0,
                      "default_left": true,
                      "node_type": "numerical_test_node",
                      "comparison_op": "<",
                      "threshold": 5.0,
                      "left_child": 1,
                      "right_child": 2
                  }, {
                      "node_id": 1,
                      "split_feature_id": 2,
                      "default_left": false,
                      "node_type": "numerical_test_node",
                      "comparison_op": "<",
                      "threshold": -3.0,
                      "left_child": 3,
                      "right_child": 4
                  }, {
                      "node_id": 2,
                      "leaf_value": 0.6000000238418579
                  }, {
                      "node_id": 3,
                      "leaf_value": -0.4000000059604645
                  }, {
                      "node_id": 4,
                      "leaf_value": 1.2000000476837159
                  }]
          }, {
              "num_nodes": 5,
              "has_categorical_split": false,
              "nodes": [{
                      "node_id": 0,
                      "split_feature_id": 1,
                      "default_left": false,
                      "node_type": "numerical_test_node",
                      "comparison_op": "<",
                      "threshold": 2.5,
                      "left_child": 1,
                      "right_child": 2
                  }, {
                      "node_id": 1,
                      "leaf_value": 1.600000023841858
                  }, {
                      "node_id": 2,
                      "split_feature_id": 2,
                      "default_left": true,
                      "node_type": "numerical_test_node",
                      "comparison_op": "<",
                      "threshold": -1.2000000476837159,
                      "left_child": 3,
                      "right_child": 4
                  }, {
                      "node_id": 3,
                      "leaf_value": 0.10000000149011612
                  }, {
                      "node_id": 4,
                      "leaf_value": -0.30000001192092898
                  }]
          }]
  }

We can also pass in some test data for prediction:

.. code-block:: python

  import numpy as np

  X = np.array(
      [
          [0.0, 0.0, -5.0],
          [0.0, 0.0, -2.0],
          [0.0, 0.0, 1.0],
          [0.0, 5.0, -5.0],
          [0.0, 5.0, -2.0],
          [0.0, 5.0, 1.0],
          [10.0, 0.0, -5.0],
          [10.0, 0.0, -2.0],
          [10.0, 0.0, 1.0],
          [10.0, 5.0, -5.0],
          [10.0, 5.0, -2.0],
          [10.0, 5.0, 1.0],
      ],
      dtype=np.float32
  )
  print(treelite.gtil.predict(model, X))

.. code-block:: none

  [[ 1.2       ]
   [ 2.8000002 ]
   [ 2.8000002 ]
   [-0.3       ]
   [ 1.3000001 ]
   [ 0.90000004]
   [ 2.2       ]
   [ 2.2       ]
   [ 2.2       ]
   [ 0.70000005]
   [ 0.70000005]
   [ 0.3       ]]

Example: Binary classifier
--------------------------
In the first example, we simply added the output of each tree to obtain the final prediction. Summing the tree outputs
is sufficient for regression models, where the target variable is a real value.

In this example, let's look at binary classifiers, where the target variable is now a binary label. We follow the
common practice, where we produce a probability score in the range of ``[0, 1]``, to indicate the relative strength for
the positive and negative classes. (Scores close to 0 indicates strong vote for the negative class; scores close to 1
indicates a strong vote for the positive class.)

To obtain probability scores, we pass the sum of the tree outputs through a **link function**
``sigmoid(x) = 1/(1+exp(-x))``. In the model builder API,
the link function is specified by the ``postprocessor`` argument. Let's look at how the builder object is constructed:

.. code-block:: python

  builder = ModelBuilder(
    threshold_type="float32",
    leaf_output_type="float32",
    metadata=Metadata(
      num_feature=3,
      task_type="kBinaryClf",
      average_tree_output=False,
      num_target=1,
      num_class=[1],
      leaf_vector_shape=(1, 1),
    ),
    # Every tree generates output for target 0, class 0
    tree_annotation=TreeAnnotation(num_tree=2, target_id=[0, 0], class_id=[0, 0]),
    # The link function for the output is the sigmoid function
    postprocessor=PostProcessorFunc(name="sigmoid"),
    # Add this value for all outputs. Also known as the intercept.
    base_scores=[0.0],
  )

Note that we've also changed ``task_type`` to ``kBinaryClf``.

Using the same definition for the two trees, we now obtain probability scores:

.. code-block:: python

  # Same tree construction logic as the first example
  # ...

  model = builder.commit()

  X = np.array(
      [
          [0.0, 0.0, -5.0],
          [0.0, 0.0, -2.0],
          [0.0, 0.0, 1.0],
          [0.0, 5.0, -5.0],
          [0.0, 5.0, -2.0],
          [0.0, 5.0, 1.0],
          [10.0, 0.0, -5.0],
          [10.0, 0.0, -2.0],
          [10.0, 0.0, 1.0],
          [10.0, 5.0, -5.0],
          [10.0, 5.0, -2.0],
          [10.0, 5.0, 1.0],
      ],
      dtype=np.float32
  )
  print(treelite.gtil.predict(model, X))

.. code-block:: none

  [[0.7685248 ]
   [0.9426758 ]
   [0.9426758 ]
   [0.4255575 ]
   [0.785835  ]
   [0.7109495 ]
   [0.90024954]
   [0.90024954]
   [0.90024954]
   [0.6681878 ]
   [0.6681878 ]
   [0.5744425 ]]

Example: multi-class classifier with vector leaf
------------------------------------------------
Now let's consider a multi-class classifier, where the target variable is now a class label whose value can be
one of ``{0, 1, 2, ..., num_class - 1}``. The tree model should now produce a 2D array of probability scores where
``score[i, k]`` represents the ``i``-th row's probability score for class ``k``.

For the sake of brevity, consider a multi-class classifier consisting of a single decision tree stump:

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph graph1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label="Feature 0 < 0.0 ?", shape=box];
      1 [label="[0.5, 0.5, 0.0]"];
      2 [label="[0.0, 0.0, 1.0]"];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="No"];
    }
  """
  Source(source, format='svg').render('../_static/builder_vecleaf1', view=False)

.. image:: ../_static/builder_vecleaf1.svg

The model has a single output target, for which there are 3 possible class labels, so we set ``num_target=1`` and
``num_class=[3]``. To indicate that the tree outputs a vector of length 3, set ``leaf_vector_shape=(1, 3)``.
The softmax function ``softmax(x) = exp(x) / sum(exp(x))`` is used as the link function, to convert the tree output
to probability scores in the range ``[0, 1]``.

.. code-block:: python

  builder = ModelBuilder(
    threshold_type="float32",
    leaf_output_type="float32",
    metadata=Metadata(
      num_feature=1,
      task_type="kMultiClf",  # To indicate multi-class classification
      average_tree_output=False,
      num_target=1,
      num_class=[3],
      leaf_vector_shape=(1, 3),
    ),
    # Every tree generates probability scores for all classes, so class_id=-1
    tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[-1]),
    # The link function for the output is the softmax function
    postprocessor=PostProcessorFunc(name="softmax"),
    # base_scores must have length (num_target * max(num_class))
    base_scores=[0.0, 0.0, 0.0],
  )

  builder.start_tree()
  builder.start_node(0)
  builder.numerical_test(
    feature_id=0,
    threshold=0.0,
    default_left=True,
    opname="<",
    left_child_key=1,
    right_child_key=2,
  )
  builder.end_node()
  builder.start_node(1)
  builder.leaf([0.5, 0.5, 0.0])
  builder.end_node()
  builder.start_node(2)
  builder.leaf([0.0, 0.0, 1.0])
  builder.end_node()
  builder.end_tree()

  model = builder.commit()
  X = np.array([[-1.0], [1.0]])
  print(treelite.gtil.predict(model, X))

.. code-block:: none

  [[0.38365173 0.38365173 0.23269653]
   [0.21194156 0.21194156 0.57611686]]

Example: multi-class classifier with scalar leaf
------------------------------------------------
It is also possible to build a multi-class classifier where each tree produces a scalar output: compute each class's
score by summing the output from a subset of decision trees. How do we know which decision tree contributes to which
class? This is where the :py:class:`~treelite.model_builder.TreeAnnotation` becomes useful.
The ``class_id`` field in :py:class:`~treelite.model_builder.TreeAnnotation` is assigned an array of integers so that
``class_id[i]`` gives the class ID to which ``i``-th tree's output counts towards. In the following example,
the outputs of Tree 0, 1, and 2 count towards Class 0, 1, and 2, respectively:


.. code-block:: python

  builder = ModelBuilder(
    threshold_type="float32",
    leaf_output_type="float32",
    metadata=Metadata(
      num_feature=1,
      task_type="kMultiClf",  # To indicate multi-class classification
      average_tree_output=False,
      num_target=1,
      num_class=[3],
      leaf_vector_shape=(1, 1),
    ),
    # Tree i produces score for class i
    tree_annotation=TreeAnnotation(
      num_tree=3,
      target_id=[0, 0, 0],
      class_id=[0, 1, 2],  # Tree i contributes towards the score of Class i
    ),
    # The link function for the output is the softmax function
    postprocessor=PostProcessorFunc(name="softmax"),
    # base_scores must have length (num_target * max(num_class))
    base_scores=[0.0, 0.0, 0.0],
  )

In this example, we will have three trees, and each tree at index ``i`` produces the score for class ``i``. In general,
we would set longer arrays for ``class_id`` to associate multiple decision trees with each class.

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph graph1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label="Feature 0 < 0.0 ?", shape=box];
      1 [label="+0.5\n(class 0)"];
      2 [label="0.0\n(class 0)"];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="No"];
    }
  """
  Source(source, format='svg').render('../_static/builder_grove_per_class1', view=False)

  source = r"""
    digraph graph1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label="Feature 0 < 0.0 ?", shape=box];
      1 [label="+0.5\n(class 1)"];
      2 [label="0.0\n(class 1)"];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="No"];
    }
  """
  Source(source, format='svg').render('../_static/builder_grove_per_class2', view=False)

  source = r"""
    digraph graph1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label="Feature 0 < 0.0 ?", shape=box];
      1 [label="0.0\n(class 2)"];
      2 [label="+1.0\n(class 2)"];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="No"];
    }
  """
  Source(source, format='svg').render('../_static/builder_grove_per_class3', view=False)

.. image:: ../_static/builder_grove_per_class1.svg

.. image:: ../_static/builder_grove_per_class2.svg

.. image:: ../_static/builder_grove_per_class3.svg

.. code-block:: python

  for tree_id in range(3):
    builder.start_tree()
    builder.start_node(0)
    builder.numerical_test(
      feature_id=0,
      threshold=0.0,
      default_left=True,
      opname="<",
      left_child_key=1,
      right_child_key=2,
    )
    builder.end_node()
    builder.start_node(1)
    builder.leaf(0.5 if tree_id < 2 else 0.0)
    builder.end_node()
    builder.start_node(2)
    builder.leaf(1.0 if tree_id == 2 else 0.0)
    builder.end_node()
    builder.end_tree()
  model = builder.commit()
  X = np.array([[-1.0], [1.0]])
  print(treelite.gtil.predict(model, X))

.. code-block:: none

  [[0.38365173 0.38365173 0.23269653]
   [0.21194156 0.21194156 0.57611686]]

Example: multi-target regressor
-------------------------------
[To be added later]
