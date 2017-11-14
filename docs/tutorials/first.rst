First tutorial
==============

**Author**: `Philip Cho <https://homes.cs.washington.edu/~chohyu01/>`_

In this tutorial, we will demonstrate the basic workflow.

.. code-block:: python

    from __future__ import absolute_import, print_function
    
    from treelite import *
    import numpy as np

Regression Example
------------------

In this tutorial, we will use a small regression example to describe the
full workflow.

Load the Boston house prices dataset
------------------------------------

Let us use the Boston house prices dataset from scikit-learn
(:py:func:`sklearn.datasets.load_boston`). It consists of 506 houses
with 13 distinct features:

.. code-block:: python

    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    print('dimensions of X = {}'.format(X.shape))
    print('dimensions of y = {}'.format(y.shape))

Train a tree ensemble model using XGBoost
-----------------------------------------

The first step is to train a tree ensemble model using XGBoost
(`dmlc/xgboost <https://github.com/dmlc/xgboost/>`_).

Disclaimer: Treelite does NOT depend on the XGBoost package in any way. 
XGBoost was used here only to provide a working example.

.. code-block:: python

    import xgboost
    dtrain = xgboost.DMatrix(X, label=y)
    params = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'reg:linear',
              'eval_metric':'rmse'}
    bst = xgboost.train(params, dtrain, 20, [(dtrain, 'train')])

Pass XGBoost model into treelite
--------------------------------

Next, we feed the trained model into treelite. If you used XGBoost to
train the model, it takes only one line of code:

.. code-block:: python

    model = Model.from_xgboost(bst)

.. note:: Using other packages to train decision trees

  With additional work, you can use models trained with other machine learning
  packages. See :doc:`this page <extern>` for instructions.

Generate shared library
-----------------------

Given a tree ensemble model, treelite will produce an **optimized
prediction subroutine** (internally represented as a C program). To use
the subroutine for prediction task, we package it as a `dynamic shared
library <https://en.wikipedia.org/wiki/Library_(computing)#Shared_libraries>`_,
which exports the prediction subroutine for other programs to use.

Before proceeding, you should decide which of the following compilers is
available on your system and set the variable ``toolchain``
appropriately:

-  ``gcc``
-  ``clang``
-  ``msvc`` (Microsoft Visual C++)

.. code-block:: python

    toolchain = 'clang'   # change this value as necessary

The choice of toolchain will be used to compile the prediction
subroutine into native code.

Now we are ready to generate the library.

.. code-block:: python

    model.export_lib(toolchain=toolchain, libpath='./mymodel.dylib', verbose=True)
                                #                            ^^^^^
                                # set correct file extension here; see the following paragraph

.. note:: File extension for shared library

  Make sure to use the correct file extension for the library,
  depending on the operating system:

  -  Windows: ``.dll``
  -  Mac OS X: ``.dylib``
  -  Linux / Other UNIX: ``.so``

.. note:: Want to deploy the model to another machine?

  This tutorial assumes that predictions will be made on the same machine that
  is running treelite. If you'd like to deploy your model to another machine
  (that may not have treelite installed), see the page :doc:`deploy`.

.. note:: Reducing compilation time for large models

  For large models, :py:meth:`~treelite.Model.export_lib` may take a long time
  to finish. To reduce compilation time, enable the ``parallel_comp`` option by
  writing

  .. code-block:: python

    model.export_lib(toolchain=toolchain, libpath='./mymodel.dylib',
                     params={'parallel_comp': 32}, verbose=True)

  which splits the prediction subroutine into 32 source files that gets compiled
  in parallel. Adjust this number according to the number of cores on your
  machine.

Use the shared library to make predictions
------------------------------------------

Once the shared library has been generated, we feed it into a separate
module (:py:mod:`treelite.runtime`) known as the runtime. The
optimized prediction subroutine is exposed through the
:py:class:`~treelite.runtime.Predictor` class:

.. code-block:: python

    from treelite.runtime import *     # runtime module
    predictor = Predictor('./mymodel.dylib', verbose=True)

We decide on which of the houses in ``X`` we should make predictions
for. Say, from 10th house to 20th:

.. code-block:: python

    batch = Batch.from_npy2d(X, rbegin=10, rend=20)

We used the method :py:meth:`~treelite.runtime.Batch.from_npy2d`
because the matrix ``X`` was a dense NumPy array (:py:class:`numpy.ndarray`).
If ``X`` were a sparse matrix (:py:class:`scipy.sparse.csr_matrix`), we would
have used the method :py:meth:`~treelite.runtime.Batch.from_csr` instead.

.. code-block:: python

    out_pred = predictor.predict(batch, verbose=True)
    print(out_pred)
