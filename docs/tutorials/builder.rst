Specifying models using model builder
=====================================

Since the scope of treelite is limited to **prediction** only, one must use
other machine learning packages to **train** decision tree ensemble models. In
this document, we will show how to import an ensemble model that had been
trained elsewhere.

**Using XGBoost or LightGBM for training?** Read :doc:`this document <extern>`
instead.

.. contents:: Contents
  :local:
  :backlinks: none

What is the model builder?
--------------------------
The :py:class:`~treelite.ModelBuilder` class is a tool used to specify decision
tree ensembles programmatically. Each tree ensemble is represented as follows:

* Each :py:class:`~treelite.ModelBuilder.Tree` object is a **dictionary** of
  nodes indexed by unique integer keys.
* A node is either a leaf node or a test node. A test node specifies its
  left and right children by their integer keys in the tree dictionary.
* Each :py:class:`~treelite.ModelBuilder` object is a **list** of
  :py:class:`~treelite.ModelBuilder.Tree` objects.

A toy example
-------------
Consider the following tree ensemble, consisting of two regression trees:

.. digraph:: toy1

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

.. digraph:: toy2

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

.. note:: Provision for missing data: default directions

  Decision trees in treelite accomodate `missing data
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

.. digraph:: toy1_1

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

.. digraph:: toy2_1

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

Next, we create a model builder object by calling the constructor for
:py:class:`~treelite.ModelBuilder`, with an ``num_feature`` argument indicating
the total number of features used in the ensemble:

.. code-block:: python

  import treelite
  builder = treelite.ModelBuilder(num_feature=3)

We also create a tree object; it will represent the first tree in the ensemble.

.. code-block:: python

  # to represent the first tree
  tree = treelite.ModelBuilder.Tree()

The first tree has five nodes, each of which is to be inserted into the tree
one at a time. The syntax for node insertion is as follows:

.. code-block:: python
  
  tree[0]   # insert a new node with key 0

Once a node has been inserted, we can refer to it by writing

.. code-block:: python
  
  tree[0]   # refer to existing node #0

The meaning of the expression ``tree[0]`` thus depends on whether the node #0
exists in the tree or not.

We may combine node insertion with a function call to specify its content.
For instance, node #0 is a test node, so we call
:py:meth:`~treelite.ModelBuilder.Node.set_numerical_test_node`:

.. code-block:: python

  # Node #0: feature 0 < 5.0 ? (default direction left)
  tree[0].set_numerical_test_node(feature_id=0,
                                  opname='<',
                                  threshold=5.0,
                                  default_left=True,
                                  left_child_key=1,
                                  right_child_key=2)

On the other hand, node #2 is a leaf node, so call
:py:meth:`~treelite.ModelBuilder.Node.set_leaf_node` instead:

.. code-block:: python

  # Node #2: leaf with output +0.6
  tree[2].set_leaf_node(0.6)

Let's go ahead and specify the other three nodes:

.. code-block:: python

  # Node #1: feature 2 < -3.0 ? (default direction right)
  tree[1].set_numerical_test_node(feature_id=2,
                                  opname='<',
                                  threshold=-3.0,
                                  default_left=False,
                                  left_child_key=3,
                                  right_child_key=4)
  # Node #3: leaf with output -0.4
  tree[3].set_leaf_node(-0.4)
  # Node #4: leaf with output +1.2
  tree[4].set_leaf_node(1.2)

We must indicate which node is the root:

.. code-block:: python
  
  # Set node #0 as root 
  tree[0].set_root()

We are now done with the first tree. We insert it with the model builder
by calling :py:meth:`~treelite.ModelBuilder.append`. (Recall that the model
builder is really a list of tree objects, hence the method name ``append``.)

.. code-block:: python

  # Insert the first tree into the ensemble
  builder.append(tree)

The second tree is constructed analogously:

.. code-block:: python

  tree2 = treelite.ModelBuilder.Tree()
  # Node #0: feature 1 < 2.5 ? (default direction right)
  tree2[0].set_numerical_test_node(feature_id=1,
                                   opname='<',
                                   threshold=2.5,
                                   default_left=False,
                                   left_child_key=1,
                                   right_child_key=2)
  # Set node #0 as root
  tree2[0].set_root()
  # Node #1: leaf with output +1.6
  tree2[1].set_leaf_node(1.6)
  # Node #2: feature 2 < -1.2 ? (default direction left)
  tree2[2].set_numerical_test_node(feature_id=2,
                                   opname='<',
                                   threshold=-1.2,
                                   default_left=True,
                                   left_child_key=3,
                                   right_child_key=4)
  # Node #3: leaf with output +0.1
  tree2[3].set_leaf_node(0.1)
  # Node #4: leaf with output -0.3
  tree2[4].set_leaf_node(-0.3)

  # Insert the second tree into the ensemble
  builder.append(tree2)

We are now done building the member trees. The last step is to call
:py:meth:`~treelite.ModelBuilder.commit` to finalize the ensemble into
a :py:class:`~treelite.Model` object:

.. code-block:: python

  # Finalize and obtain Model object
  model = builder.commit()

.. note:: Difference between :py:class:`~treelite.ModelBuilder` and
  :py:class:`~treelite.Model` objects

  Why does treelite require one last step of "committing"? All
  :py:class:`~treelite.Model` objects are **immutable**; once constructed,
  they cannot be modified at all. So you won't be able to add a tree or a node
  to an existing :py:class:`~treelite.Model` object, for instance. On the other
  hand, :py:class:`~treelite.ModelBuilder` objects are mutable, so that you
  can iteratively build trees.

To ensure we got all details right, we can examine the resulting C program.

.. code-block:: python

  model.compile(dirpath='./test')
  with open('./test/test.c', 'r') as f:
    for line in f.readlines():
      print(line, end='')

which produces the output

.. code-block:: c

  /* Other functions omitted for space consideration */
  float predict_margin(union Entry* data) {
    float sum = 0.0f;
    if (!(data[0].missing != -1) || data[0].fvalue < 5) {
      if ( (data[2].missing != -1) && data[2].fvalue < -3) {
        sum += (float)-0.4;
      } else {
        sum += (float)1.2;
      }
    } else {
      sum += (float)0.6;
    }
    if ( (data[1].missing != -1) && data[1].fvalue < 2.5) {
      sum += (float)1.6;
    } else {
      if (!(data[2].missing != -1) || data[2].fvalue < -1.2) {
        sum += (float)0.1;
      } else {
        sum += (float)-0.3;
      }
    }
    return sum;
  }

The toy example has been helpful as an illustration, but it is impractical
to manually specify nodes for real-world ensemble models. The following section
will show us how to automate the tree building process. We will look at
scikit-learn in particular.

Using the model builder to interface with scikit-learn
------------------------------------------------------
**Scikit-learn** (`scikit-learn/scikit-learn
<https://github.com/scikit-learn/scikit-learn>`_) is a Python machine learning
package known for its versatility and ease of use. It supports a wide variety
of models and algorithms. 

Treelite will be able to work with any decision tree ensemble models produced
by scikit-learn. In particular, it will be able to work with

* :py:class:`sklearn.ensemble.GradientBoostingClassifier`
* :py:class:`sklearn.ensemble.GradientBoostingRegressor`
* :py:class:`sklearn.ensemble.RandomForestClassifier`
* :py:class:`sklearn.ensemble.RandomForestRegressor`

.. note:: Why scikit-learn? How about other packages?

  We had to pick a specific example for programmatic tree construction, so we
  chose scikit-learn. If you're using another package, don't lose heart. As you
  read through the rest of section, notice how specific pieces of information
  about the tree ensemble model are being extracted. As long as your choice
  of package exposes equivalent information, you'll be able to adapt the example
  to your needs.

.. note:: Adaboost ensembles not yet supported

  Treelite currently does not support weighting of member trees, so you won't
  be able to use Adaboost ensembles.

Regression with RandomForestRegressor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start with the `Boston house prices dataset <https://archive.ics.uci.edu/
ml/machine-learning-databases/housing/housing.names>`_, a regression problem.
(Classification problems are somewhat trickier, so we'll save them for later.)

We'll be using :py:class:`~sklearn.ensemble.RandomForestRegressor`, a random
forest for regression. A **random forest** is an ensemble of decision trees
that are independently trained on random samples from the training data. See
`this page
<http://scikit-learn.org/stable/modules/ensemble.html#random-forests>`_ for
more details. For now, just remember to specify ``random_forest=True`` in the
:py:class:`~treelite.ModelBuilder` constructor.

.. code-block:: python

  import sklearn.datasets
  import sklearn.ensemble
  # Load the Boston housing dataset
  X, y = sklearn.datasets.load_boston(return_X_y=True)
  # Train a random forest regressor with 10 trees
  clf = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
  clf.fit(X, y)

We shall programmatically construct :py:class:`~treelite.ModelBuilder.Tree`
objects from internal attributes of the scikit-learn model. We only need
to define a few helper functions.

For the rest of sections, we'll be diving into lots of details that are specific
to scikit-learn. Many details have been adopted from `this reference page <http:
//scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html>`_.

**The function process_model()** takes in a scikit-learn ensemble object and
returns the completed :py:class:`~treelite.Model` object:

.. code-block:: python

  def process_model(sklearn_model):
    # Initialize treelite model builder
    # Set random_forest=True for random forests
    builder = treelite.ModelBuilder(num_feature=sklearn_model.n_features_,
                                    random_forest=True)

    # Iterate over individual trees
    for i in range(sklearn_model.n_estimators):
      # Process the i-th tree and add to the builder
      builder.append( process_tree(sklearn_model.estimators_[i].tree_) )
    
    return builder.commit()

The usage of this function is as follows:

.. code-block:: python

  model = process_model(clf)

We won't have space here to discuss all internals of scikit-learn objects, but
a few details should be noted:

* The attribute ``n_features_`` stores the number of features used anywhere
  in the tree ensemble.
* The attribute ``n_estimators`` stores the number of member trees.
* The attribute ``estimators_`` is an array of handles that store the individual
  member trees. To access the object for the ``i``-th tree, write
  ``estimators_[i].tree_``. This object will be passed to the function
  ``process_tree()``.

**The function process_tree()** takes in a single scikit-learn tree object
and returns an object of type :py:class:`~treelite.ModelBuilder.Tree`:

.. code-block:: python

  def process_tree(sklearn_tree):
    treelite_tree = treelite.ModelBuilder.Tree()
    # Node #0 is always root for scikit-learn decision trees
    treelite_tree[0].set_root()

    # Iterate over each node: node ID ranges from 0 to [node_count]-1
    for node_id in range(sklearn_tree.node_count):
      process_node(treelite_tree, sklearn_tree, node_id)

    return treelite_tree

Explanations:

* The attribute ``node_count`` stores the number of nodes in the decision tree.
* Each node in the tree has a unique ID ranging from 0 to ``[node_count]-1``.

**The function process_node()** determines whether each node is a leaf node
or a test node. It does so by looking at the attribute ``children_left``:
If the left child of the node is set to -1, that node is thought to be
a leaf node.

.. code-block:: python

  def process_node(treelite_tree, sklearn_tree, node_id):
    if sklearn_tree.children_left[node_id] == -1:  # leaf node
      process_leaf_node(treelite_tree, sklearn_tree, node_id)
    else:                                          # test node
      process_test_node(treelite_tree, sklearn_tree, node_id)

**The function process_test_node()** extracts the content of a test node
and passes it to the :py:class:`~treelite.ModelBuilder.Tree` object that is
being constructed.

.. code-block:: python

  def process_test_node(treelite_tree, sklearn_tree, node_id):
    # Initialize the test node with given node ID
    treelite_tree[node_id].set_numerical_test_node(
                          feature_id=sklearn_tree.feature[node_id],
                          opname='<=',
                          threshold=sklearn_tree.threshold[node_id],
                          default_left=True,            # see note
                          left_child_key=sklearn_tree.children_left[node_id],
                          right_child_key=sklearn_tree.children_right[node_id])

Explanations:

* The attribute ``feature`` is the array containing feature indices used
  in test nodes.
* The attribute ``threshold`` is the array containing threshold values used
  in test nodes.
* All tests are in the form of ``[feature value] <= [threshold]``.
* The attributes ``children_left`` and ``children_right`` together store
  children's IDs for test nodes.

.. note:: Scikit-learn and missing data

  Scikit-learn handles missing data differently than XGBoost and treelite.
  Before training an ensemble model, you'll have to `impute
  <http://scikit-learn.org/stable/modules/preprocessing.html#imputation>`_
  missing values. For this reason, test nodes in scikit-learn tree models will
  contain no "default direction." We will assign ``default_left=True``
  arbitrarily for test nodes to keep treelite happy.

**The function process_leaf_node()** defines a leaf node:

.. code-block:: python

  def process_leaf_node(treelite_tree, sklearn_tree, node_id):
    # the `value` attribute stores the output for every leaf node.
    leaf_value = sklearn_tree.value[node_id].squeeze()
    # Initialize the leaf node with given node ID
    treelite_tree[node_id].set_leaf_node(leaf_value)

Let's test it out:

.. code-block:: python

  model = process_model(clf)
  model.export_lib(libpath='./libtest.dylib', toolchain='gcc', verbose=True)

  import treelite.runtime
  predictor = treelite.runtime.Predictor(libpath='./libtest.dylib')
  predictor.predict(treelite.runtime.Batch.from_npy2d(X))

Regression with GradientBoostingRegressor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Gradient boosting** is an algorithm where decision trees are trained one at a
time, ensuring that latter trees complement former trees. See `this page
<http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting>`_
for more details. Treelite makes distinction between random forests and
gradient boosted trees by the value of ``random_forest`` flag in the
:py:class:`~treelite.ModelBuilder` constructor.

.. note:: Set ``init='zero'`` to ensure compatibility

  To make sure that the gradient boosted model is compatible with treelite,
  make sure to set ``init='zero'`` in the
  :py:class:`~sklearn.ensemble.GradientBoostingRegressor` constructor. This
  ensures that the compiled prediction subroutine will produce the correct
  prediction output.

.. code-block:: python

  # Gradient boosting regressor
  # Notice the argument init='zero'
  clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=10,
                                                   init='zero')
  clf.fit(X, y)

We will recycle most of the helper code we wrote earlier. Only two functions
will need to be modified:

.. code-block:: python

  def process_model(sklearn_model):
    # Initialize treelite model builder
    # Set random_forest=False for gradient boosted trees
    builder = treelite.ModelBuilder(num_feature=sklearn_model.n_features_,
                                    random_forest=False)
    for i in range(sklearn_model.n_estimators):
      # Process i-th tree and add to the builder
      builder.append( process_tree(sklearn_model.estimators_[i][0].tree_) )

    return builder.commit()

  def process_leaf_node(treelite_tree, sklearn_tree, node_id):
    # Need to shrink each leaf output by the learning rate
    leaf_value = clf.learning_rate * sklearn_tree.value[node_id].squeeze()
    # Initialize the leaf node with given node ID
    treelite_tree[node_id].set_leaf_node(leaf_value)

Some details specific to :py:class:`~sklearn.ensemble.GradientBoostingRegressor`:

* To indicate the use of gradient boosting (as opposed to random forests), we
  set ``random_forest=False`` in the :py:class:`~treelite.ModelBuilder`
  constructor.
* Each tree object is now accessed with the expression
  ``estimators_[i][0].tree_``, as ``estimators_[i]`` returns an array consisting
  of a single tree (``i``-th tree).
* Each leaf output in gradient boosted trees are "unscaled": it needs to be
  scaled by the learning rate.

Let's test it:

.. code-block:: python

  # Convert to treelite model
  model = process_model(clf)
  # Generate shared library
  model.export_lib(libpath='./libtest2.dylib', toolchain='gcc', verbose=True)
  # Make prediction with predictor
  predictor = treelite.runtime.Predictor(libpath='./libtest2.dylib')
  predictor.predict(treelite.runtime.Batch.from_npy2d(X))

Binary Classification with RandomForestClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For binary classification, let's use `the digits dataset
<http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html>`_.
We will take 0's and 1's from the dataset and treat 0's as the negative class
and 1's as the positive.

.. code-block:: python

  # load a binary classification problem
  # set n_class=2 to produce two classes
  digits = sklearn.datasets.load_digits(n_class=2)
  X, y = digits['data'], digits['target']
  # should print [0 1]
  print(np.unique(y))
  
  # train a random forest classifier
  clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
  clf.fit(X, y)

Random forest classifiers in scikit-learn store **frequency counts** for the
positive and negative class. For instance, a leaf node may output a set of
counts

.. code-block:: none

  [ 100, 200 ]

which indicates the following:

* 300 data points in the training set "belong" to this leaf node, in the sense
  that they all satisfy the precise sequence of conditions leading to that
  particular leaf node. The picture below shows that each leaf node represents
  a unique sequence of conditions: 

.. digraph:: leaf_count_illustration

  graph [fontname = "helvetica"];
  node [fontname = "helvetica"];
  edge [fontname = "helvetica"];
  0 [label="Feature X < x.x ?", shape=box, color=red, fontcolor=red];
  1 [label="Feature X < x.x ?", shape=box];
  2 [label="Feature X < x.x ?", shape=box, color=red, fontcolor=red];
  3 [label="...", shape=none];
  4 [label="...", shape=none];
  5 [label="...", shape=none, fontcolor=red];
  6 [label="...", shape=none];
  7 [label="...", shape=none];
  8 [label="Leaf node", color=red, fontcolor=red, fontname = "helvetica bold"];
  0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
  0 -> 2 [labeldistance=2.0, labelangle=-45,
          headlabel="No", color=red, fontcolor=red];
  1 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
  1 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
  2 -> 5 [labeldistance=2.0, labelangle=45, headlabel="Yes",
          color=red, fontcolor=red];
  2 -> 6 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
  5 -> 7;
  5 -> 8 [color=red];

* 100 of them are labeled negative; and
* the remaining 200 are labeled positive.

Again, most of the helper functions may be re-used; only two functions need to
be rewritten. Explanation will follow after the code:

.. code-block:: python

  def process_model(sklearn_model):
    builder = treelite.ModelBuilder(num_feature=sklearn_model.n_features_,
                                    random_forest=True)
    for i in range(sklearn_model.n_estimators):
      # Process i-th tree and add to the builder
      builder.append( process_tree(sklearn_model.estimators_[i].tree_) )
  
    return builder.commit()
  
  def process_leaf_node(treelite_tree, sklearn_tree, node_id):
    # Get counts for each label (+/-) at this leaf node
    leaf_count = sklearn_tree.value[node_id].squeeze()
    # Compute the fraction of positive data points at this leaf node
    fraction_positive = float(leaf_count[1]) / leaf_count.sum()
    # The fraction above is now the leaf output
    treelite_tree[node_id].set_leaf_node(fraction_positive)

As noted earlier, we access the frequency counts at each leaf node, reading the 
``value`` attribute of each tree. Then we compute the fraction of positive
data points with respect to all training data points belonging to the leaf.
This fraction then becomes the leaf output. This way, leaf nodes now produce
single numbers rather than frequency count arrays.

Why did we have to compute a fraction? **For binary classification,
treelite expects each tree to produce a single number output.** At prediction
time, the outputs from the member trees will get **averaged** to produce the
final prediction, which is also a single number. By setting the positive
fraction as the leaf output, we ensure that the final prediction is a proper
probability value. For instance, if an ensemble consisting of 5 trees produces
the following set of outputs

.. code-block:: none

  [ 0.1, 0.7, 0.4, 0.3, 0.7 ]

then the final prediction will be 0.44, which we interpret as 44% confidence
for the positive class.

Multi-class Classification with RandomForestClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let's use `the digits dataset
<http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html>`_
again, this time with 4 classes (i.e. 0's, 1's, 2's, and 3's).

.. code-block:: python

  # load a multi-class classification problem
  # set n_class=4 to produce four classes
  digits = sklearn.datasets.load_digits(n_class=4)
  X, y = digits['data'], digits['target']
  # should print [0 1 2 3]
  print(np.unique(y))
  
  # train a random forest classifier
  clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
  clf.fit(X, y)

Random forest classifiers in scikit-learn store frequency counts (see the
explanation in the previous section). For instance, a leaf node may output a
set of counts

.. code-block:: none

  [ 100, 400, 300, 200 ]

which shows that the total of 1000 training data points belong to this leaf node
and that 100, 400, 300, and 200 of them are labeled class 0, 1, 2, and 3,
respectively.

We will have to re-write the **process_leaf_node()** function to accomodate
multiple classes.

.. code-block:: python

  def process_model(sklearn_model):
    # must specify num_output_group and pred_transform
    builder = treelite.ModelBuilder(num_feature=sklearn_model.n_features_,
                                    num_output_group=sklearn_model.n_classes_,
                                    random_forest=True,
                                    params={'pred_transform':'identity_multiclass'})
    for i in range(sklearn_model.n_estimators):
      # Process i-th tree and add to the builder
      builder.append( process_tree(sklearn_model.estimators_[i].tree_) )
  
    return builder.commit()
  
  def process_leaf_node(treelite_tree, sklearn_tree, node_id):
    # Get counts for each label class (0, 1, 2, 3) at this leaf node
    leaf_count = sklearn_tree.value[node_id].squeeze()
    # Compute the probability distribution over label classes
    prob_distribution = leaf_count / leaf_count.sum()
    # The leaf output is the probability distribution
    treelite_tree[node_id].set_leaf_node(prob_distribution)

The process_leaf_node() function is quite similar to what we had for the binary
classification case. Only different is that, instead of computing the fraction
of the positive class, we compute the **probability distribution** for all
possible classes. Each leaf node thus will store the probability distribution
of possible class outcomes.

The process_model() function is also similar to what we had before. The crucial
difference is the existence of parameters ``num_output_group`` and
``pred_transform``. The ``num_output_group`` parameter is used only for
multi-class classification: it should store the number of classes (in this
example, 4). The ``pred_transform`` parameter, which is tucked into the
``params`` dictionary, should be set to ``'identity_multiclass'``, to indicate
that the prediction should be made simply by averaging the probability
distribution outputed by each leaf node. For instance, if an ensemble 
consisting of 3 trees produces the following set of outputs

.. code-block:: none

  [ [ 0.5, 0.5, 0.0 ], [ 0.1, 0.6, 0.3 ], [ 0.2, 0.5, 0.3 ] ]

then the final prediction will be the average
``[ 0.26666667, 0.53333333, 0.2 ]``, which indicates 26.7% probability for the
first class, 53.3% for the second, and 20.0% for the third.

Binary Classification with GradientBoostingClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**[COMING SOON]**

Multi-class Classification with GradientBoostingClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**[COMING SOON]**
