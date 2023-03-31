Specifying models using JSON string
===================================

Treelite supports loading models from major tree libraries, such as XGBoost and
scikit-learn. However, you may want to use models trained by other tree
libraries that are not directly supported by Treelite. The JSON importer is
useful in this use case. (Alternatively, consider
:doc:`using the model builder <builder>` instead.)

.. note:: :py:meth:`~treelite.Model.import_from_json` is strict about which JSON strings to accept

  Some tree libraries such as XGBoost, Catboost, and cuML RandomForest let users to dump tree
  models as JSON strings. However, :py:meth:`~treelite.Model.import_from_json` will not accept
  these strings. It requires a particular set of fields, as outlined in the tutorial below.
  Here are the suggested methods for converting your tree model into Treelite format:
  
  1. If you are using XGBoost, LightGBM, or scikit-learn, use methods
     :py:meth:`~treelite.Model.load` and :py:meth:`~treelite.sklearn.import_model`.
  2. If you are using cuML RandomForest, convert the model directly to Treelite objects as follows:
  
  .. code:: python
  
    cuml_rf_model.convert_to_treelite_model().to_treelite_checkpoint("checkpoint.bin")
    tl_model = treelite.Model.deserialize("checkpoint.bin")
  
  3. If you are using Catboost or other tree libraries that Treelite do not support directly,
     write a custom program to format your tree models to produce the correctly formatted JSON
     string. Make sure that all fields are put in place, such as ``task_param``, ``model_param``
     and others.

.. contents:: Contents
  :local:

Toy Example
-----------

Consider the following tree ensemble, consisting of two regression trees:

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph toy1 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label=<<FONT COLOR="red">0:</FONT> Feature 1 ∈ {1, 2, 4} ?>, shape=box];
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
  Source(source, format='png').render('../_static/json_import_toy1', view=False)
  Source(source, format='svg').render('../_static/json_import_toy1', view=False)

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph toy2 {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0 [label=<<FONT COLOR="red">1:</FONT> Feature 0 &lt; 2.5 ?>, shape=box];
      1 [label=<<FONT COLOR="red">2:</FONT> +1.6>];
      2 [label=<<FONT COLOR="red">4:</FONT> Feature 2 &lt; -1.2 ?>, shape=box];
      3 [label=<<FONT COLOR="red">6:</FONT> +0.1>];
      4 [label=<<FONT COLOR="red">8:</FONT> -0.3>];
      0 -> 1 [labeldistance=2.0, labelangle=45, headlabel="Yes"];
      0 -> 2 [labeldistance=2.0, labelangle=-45, headlabel="           No/Missing"];
      2 -> 3 [labeldistance=2.0, labelangle=45, headlabel="Yes/Missing           "];
      2 -> 4 [labeldistance=2.0, labelangle=-45, headlabel="No"];
    }
  """
  Source(source, format='png').render('../_static/json_import_toy2', view=False)
  Source(source, format='svg').render('../_static/json_import_toy2', view=False)

.. raw:: html

  <p>
  <img src="../_static/json_import_toy1.svg"
       onerror="this.src='../_static/json_import_toy1.png'; this.onerror=null;">
  <img src="../_static/json_import_toy2.svg"
       onerror="this.src='../_static/json_import_toy2.png'; this.onerror=null;">
  </p>

.. role:: red

where each node is assign a **unique integer key**, indicated in :red:`red`.
Note that integer keys need to be unique only within the same tree.

You can construct this tree ensemble by calling
:py:meth:`~treelite.Model.import_from_json` with an appropriately formatted
JSON string. We will give you the example code first; in the following section,
we will explain the meaining of each field in the JSON string.

.. note:: :py:meth:`~treelite.Model.dump_as_json` will NOT preserve the JSON string that's passed into :py:meth:`~treelite.Model.import_from_json`

  The operation performed in :py:meth:`~treelite.Model.import_from_json` is strictly one-way.
  So the output of :py:meth:`~treelite.Model.dump_as_json` will differ from the JSON string
  you used in calling :py:meth:`~treelite.Model.import_from_json`.

.. code-block:: python
  :linenos:
  :emphasize-lines: 78

  import treelite

  json_str = """
  {
      "num_feature": 3,
      "task_type": "kBinaryClfRegr",
      "average_tree_output": false,
      "task_param": {
          "output_type": "float",
          "grove_per_class": false,
          "num_class": 1,
          "leaf_vector_size": 1
      },
      "model_param": {
          "pred_transform": "identity",
          "global_bias": 0.0
      },
      "trees": [
          {
              "root_id": 0,
              "nodes": [
                  {
                      "node_id": 0,
                      "split_feature_id": 1,
                      "default_left": true,
                      "split_type": "categorical",
                      "categories_list": [1, 2, 4],
                      "categories_list_right_child": false,
                      "left_child": 1,
                      "right_child": 2
                  },
                  {
                      "node_id": 1,
                      "split_feature_id": 2,
                      "default_left": false,
                      "split_type": "numerical",
                      "comparison_op": "<",
                      "threshold": -3.0,
                      "left_child": 3,
                      "right_child": 4
                  },
                  {"node_id": 2, "leaf_value": 0.6},
                  {"node_id": 3, "leaf_value": -0.4},
                  {"node_id": 4, "leaf_value": 1.2}
              ]
          },
          {
              "root_id": 1,
              "nodes": [
                  {
                      "node_id": 1,
                      "split_feature_id": 0,
                      "default_left": false,
                      "split_type": "numerical",
                      "comparison_op": "<",
                      "threshold": 2.5,
                      "left_child": 2,
                      "right_child": 4
                  },
                  {
                      "node_id": 4,
                      "split_feature_id": 2,
                      "default_left": true,
                      "split_type": "numerical",
                      "comparison_op": "<",
                      "threshold": -1.2,
                      "left_child": 6,
                      "right_child": 8
                  },
                  {"node_id": 2, "leaf_value": 1.6},
                  {"node_id": 6, "leaf_value": 0.1},
                  {"node_id": 8, "leaf_value": -0.3}
              ]
          }
      ]
  }
  """
  model = treelite.Model.import_from_json(json_str)


Building model components using JSON
------------------------------------

Model metadata
^^^^^^^^^^^^^^
In the beginning, we must specify certain metadata of the model.

* ``num_teature``: Number of features (columns) in the training data
* ``average_tree_output``: Whether to average the outputs of trees. Set this to
  True if the model is a random forest.
* ``task_type`` / ``task_param``: :ref:`Parameters that together define a
  machine learning task <task_param>`.
* ``model_param``: :ref:`Other important parameters in the model <model_param>`.

.. _task_param:

Task Parameters: Define a machine learing task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``task_type`` parameter is closely related to the content of ``task_param``.
The ``task_param`` object has the following parameters:

* ``output_type``: Type of leaf output. Either ``float`` or ``int``.
* ``grove_per_class``: Boolean indicating a particular organization of multi-class
  classifier.
* ``num_class``: Number of targer classes in a multi-class classifier. Set this
  to 1 if the model is a binary classifier or a non-classifier.
* ``leaf_vector_size``: Length of leaf output. A value of 1 indicates scalar output.

The docstring of :cpp:enum:`TaskType` explains the relationship between
``task_type`` and the parameters in ``task_param``:

.. doxygenenum:: TaskType
  :project: treelite

.. _model_param:

Other Model Parameters
~~~~~~~~~~~~~~~~~~~~~~
The ``model_param`` field contains the parameters described in :doc:`../knobs/model_param`.
You may safely omit a parameter as long as it has a default value.

Tree nodes
^^^^^^^^^^
Each tree object must have ``root_id`` field to indicate which node is the root node.

The ``nodes`` array must have node objects. Each node object must have ``node_id`` field.
It will also have other fields, depending on the type of the node. A typical leaf node
will be like this:

.. code-block:: json

  {"node_id": 2, "leaf_value": 0.6}

To output a leaf vector, use a list instead.

.. code-block:: json

  {"node_id": 2, "leaf_value": [0.6, 0.4]}

A typical internal node with numerical test:

.. code-block:: json

  {
      "node_id": 1,
      "split_feature_id": 2,
      "default_left": false,
      "split_type": "numerical",
      "comparison_op": "<",
      "threshold": -3.0,
      "left_child": 3,
      "right_child": 4
  }

A typical internal node with categorical test:

.. code-block:: json

  {
      "node_id": 0,
      "split_feature_id": 1,
      "default_left": true,
      "split_type": "categorical",
      "categories_list": [1, 2, 4],
      "categories_list_right_child": false,
      "left_child": 1,
      "right_child": 2
  }

For the categorical test, the test criterion is in the form of

.. code-block:: none

  [Feature value] \in [categories_list]

where the ``categories_list`` defines a (mathematical) set.
When the test criteron is evaluated to be true, the prediction function
traverses to the left child node (if ``categories_list_right_child=false``)
or to the right child node (if ``categories_list_right_child=true``).
