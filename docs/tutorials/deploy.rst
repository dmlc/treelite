Deploying models
================

After all the hard work you did to train your tree ensemble model, you now have
to **deploy** the model. Deployment refers to distributing your model to
other machines and devices so as to make predictions on them. To facilitate
the coming discussions, let us define a few terms.

* **Host machine** : the machine running Treelite.
* **Target machine** : the machine on which predictions will be made. The host
  machine may or may not be identical to the target machine. In cases where
  it's infeasible to install Treelite on the target machine, the host and
  target machines will be necessarily distinct.
* **Shared library** : a blob of executable subroutines that can be imported by
  other native applications. Shared libraries will often have file extensions
  .dll, .so, or .dylib. Going back to the particular context of tree deployment,
  Treelite will produce a shared library containing the prediction subroutine
  (compiled to native machine code).
* **Runtime package** : a :doc:`tiny fraction<../treelite-runtime-api>` of the
  full Treelite package, consisting of a few helper functions that lets you
  easily load shared libraries and make predictions. The runtime is good to
  have, but on systems lacking Python we can do without it.

In this document, we will document two options for deployment. We will present the programming
interface each deployment option presents, as well as its dependencies and requirements.

.. contents:: Contents
  :local:
  :backlinks: none
  :depth: 2

Option 1: Deploy prediction code with the runtime package
---------------------------------------------------------
If feasible, this option is probably the most convenient. On the target machine, install the
Treelite runtime by running pip:

.. code-block:: bash

  python3 -m pip install treelite_runtime --user

Once the Treelite runtime is installed, it suffices to follow instructions in :doc:`first`.

Option 2: Deploy prediciton code only
-------------------------------------

With this option, neither Python nor a C++ compiler is required. You should be
able to adopt this option using any basic installation of UNIX-like operating systems.

Dependencies and Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The target machine shall meet the following conditions:

* A C compiler is available.
* The C compiler supports the following features of the
  `C99 standard <https://en.wikipedia.org/wiki/C99>`_: inline functions;
  declaration of loop variables inside ``for`` loop; the ``expf`` function in
  ``<math.h>``; the ``<stdint.h>`` header.
* GNU Make or Microsoft NMake is installed.
* An archive utility exists that can open a .zip archive.

Deployment instructions
^^^^^^^^^^^^^^^^^^^^^^^
\1. On the host machine, install Treelite and import your tree ensemble model.
You should end up with the model object of type :py:class:`~treelite.Model`.

.. code-block:: python

  ### Run this block on the **host** machine

  import treelite
  model = treelite.Model.load('your_model.model', 'xgboost')
  # You may also use `from_xgboost` method or the builder class

\2. Export your model as a **source package** by calling the method
:py:meth:`~treelite.Model.export_srcpkg` of the :py:class:`~treelite.Model`
object. The source package will contain C code representation of the prediction
subroutine.

.. code-block:: python

  ### Continued from the previous code block

  # Operating system of the target machine
  platform = 'unix'
  # C compiler to use to compile prediction code on the target machine
  toolchain = 'gcc'
  # Save the source package as a zip archive named mymodel.zip
  # Later, we'll use this package to produce the library mymodel.so.
  model.export_srcpkg(platform=platform, toolchain=toolchain,
                      pkgpath='./mymodel.zip', libname='mymodel.so',
                      verbose=True)

.. note:: On the value of ``toolchain``

  Treelite supports only three toolchain configurations ('msvc', 'gcc', 'clang')
  for which it generates Makefiles. If you are using a compiler other than
  these three, you will have to write your own Makefile. For now, just set
  ``toolchain='gcc'`` and move on.

After calling :py:meth:`~treelite.Model.export_srcpkg`, you should be able to
find the zip archive named ``mymodel.zip`` inside the current working directory.

.. code-block:: console

  john.doe@host-machine:/home/john.doe/$ ls .
  mymodel.zip   your_model.model

The content of ``mymodel.zip`` consists of the header and source files, as well
as the Makefile:

.. code-block:: console

  john.doe@host-machine:/home/john.doe/$ unzip -l mymodel.zip
  Archive:  mymodel.zip
    Length      Date    Time    Name
  ---------  ---------- -----   ----
          0  11-01-2017 23:11   mymodel/
        167  11-01-2017 23:11   mymodel/Makefile
    4831036  11-01-2017 23:11   mymodel/mymodel.c
        311  11-01-2017 23:11   mymodel/mymodel.h
        109  11-01-2017 23:11   mymodel/recipe.json
  ---------                     -------
    4831623                     5 files

\3. Now you are ready to deploy the model to the target machine. Copy to the
target machine the archive ``mymodel.zip`` (source package).

.. code-block:: console

  john.doe@host-machine:/home/john.doe/$ sftp john.doe@target-machine
  Connected to target-machine.
  sftp> put mymodel.zip
  Uploading mymodel.zip to /home/john.doe/mymodel.zip
  mymodel.zip                             100%  410KB 618.2KB/s   00:00
  sftp> quit

\4. It is time to move to the target machine. On the target machine, extract
the archive ``mymodel.zip``:

.. code-block:: console

  john.doe@host-machine:/home/john.doe/$ ssh john.doe@target-machine
  Last login: Tue Oct 31 00:43:36 2017 from host-machine

  john.doe@target-machine:/home/john.doe/$ unzip mymodel.zip
  Archive:  mymodel.zip
     creating: mymodel/
    inflating: mymodel/Makefile
    inflating: mymodel/mymodel.c
    inflating: mymodel/mymodel.h
    inflating: mymodel/recipe.json

\5. Build the source package (using GNU Make or NMake).

.. code-block:: console

  john.doe@target-machine:/home/john.doe/$ cd mymodel
  john.doe@target-machine:/home/john.doe/mymodel/$ make
  gcc -c -O3 -o mymodel.o mymodel.c -fPIC -std=c99 -flto -fopenmp
  gcc -shared -O3 -o mymodel.so mymodel.o -std=c99 -flto -fopenmp
  john.doe@target-machine:/home/john.doe/mymodel/$ ls
  Makefile       mymodel.c      mymodel.so
  mymodel.h      mymodel.o      recipe.json

.. note:: Parallel compilation with GNU Make

  If you used ``parallel_comp`` option to split the model into multiple source
  files, you can take advantage of parallel compilation. Simply replace ``make``
  with ``make -jN``, where ``N`` is replaced with the number of workers to
  launch. Setting ``N`` too high may result into memory shortage.

.. note:: Using other compilers

  If you are using a compiler other than gcc, clang, or Microsoft Visual C++,
  you will need to compose your own Makefile. Open the ``Makefile`` and
  make necessary changes.

Prediction instructions
^^^^^^^^^^^^^^^^^^^^^^^
The prediction library provides the function ``predict`` with the
following signature:

.. code-block:: c

  float predict(union Entry* data, int pred_margin);

Here, the argument ``data`` must be an array of length ``M``, where ``M`` is
the number of features used in the tree ensemble. The ``data`` array stores
all the feature values of a single row. To indicate presence or absence of
a feature value, we use the union type ``Entry``, which defined as

.. code-block:: c

  union Entry {
    int missing;
    float fvalue;
  };

For missing values, we set the ``missing`` field to -1. For non-missing ones, we
set the ``fvalue`` field to the feature value. The total number of features
is given by the function

.. code-block:: c

  size_t get_num_feature(void);

Let's look at an example. We'd start by initializing the array ``inst``, a dense
aray to hold feature values of a single data row:

.. code-block:: c

  /* number of features */
  const size_t num_feature = get_num_feature();
  /* inst: dense vector storing feature values */
  union Entry* inst = malloc(sizeof(union Entry) * num_feature);
  /* clear inst with all missing values */
  for (i = 0; i < num_feature; ++i) {
    inst[i].missing = -1;
  }

Before calling the function ``predict``, the array ``inst`` needs to be
initialized with missing and present feature values. The following peudocode
illustrates the idea:

.. code-block:: none

  For each data row rid:
    inst[i].missing == -1 for every i, assuming all features lack values

    For each feature i for which the data row in fact has a feature value:
      Set inst[i].fvalue = [feature value], to indicate presence

    Call predict(inst, 0) and get prediction for the data row rid

    For each feature i for which the row has a feature value:
      Set inst[i].missing = -1, to prepare for next row (rid + 1)

The task is not too difficult as long as the input data is given as a particular
form of sparse matrix: the `Compressed Sparse Row\
<http://www.netlib.org/utk/people/JackDongarra/etemplates/node373.html>`_ format.
The sparse matrix consists of three arrays:

* ``val`` stores nonzero entries in
  `row-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_.
* ``col_ind`` stores column indices of the entries in ``val``. The expression
  ``col_ind[i]`` indicates the column index of the ``i`` th entry ``val[i]``.
* ``row_ptr`` stores the locations in ``val`` that start and end data rows. The
  ``i`` th data row is given by the array slice ``val[row_ptr[i]:row_ptr[i+1]]``.

.. code-block:: c

  /* nrow : number of data rows */
  for (rid = 0; rid < nrow; ++rid) {
    ibegin = row_ptr[rid];
    iend = row_ptr[rid + 1];
    /* Fill nonzeros */
    for (i = ibegin; i < iend; ++i) {
      inst[col_ind[i]].fvalue = val[i];
    }
    out_pred[rid] = predict(inst, 0);
    /* Drop nonzeros */
    for (i = ibegin; i < iend; ++i) {
      inst[col_ind[i]].missing = -1;
    }
  }

It only remains to create three arrays ``val``, ``col_ind``, and ``row_ptr``.
You may want to use a third-pary library here to read from
a SVMLight format. For now, we'll punt the issue of loading the input data
and write it out as constants in the program:

.. code-block:: c

  #include <stdio.h>
  #include <stdlib.h>
  #include "mymodel.h"

  int main(void) {
    /* 5x13 "sparse" matrix, in CSR format
       [[ 0.  ,  0.  ,  0.68,  0.99,  0.  ,  0.11,  0.  ,  0.82,  0.  ,
          0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.99,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
          0.61,  0.  ,  0.  ,  0.  ],
        [ 0.02,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
          0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.36,  0.  ,  0.82,  0.  ,  0.  ,  0.57,  0.  ,
          0.  ,  0.  ,  0.  ,  0.75],
        [ 0.47,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
          0.  ,  0.  ,  0.45,  0.  ]]
    */
    const float val[] = {0.68, 0.99, 0.11, 0.82, 0.99, 0.61, 0.02, 0.36, 0.82,
                         0.57, 0.75, 0.47, 0.45};
    const size_t col_ind[] = {2, 3, 5, 7, 2, 9, 0, 2, 4, 7, 12, 0, 11};
    const size_t row_ptr[] = {0, 4, 6, 7, 11, 13};
    const size_t nrow = 5;
    const size_t ncol = 13;

    /* number of features */
    const size_t num_feature = get_num_feature();
    /* inst: dense vector storing feature values */
    union Entry* inst = malloc(sizeof(union Entry) * num_feature);
    float* out_pred = malloc(sizeof(float) * nrow);
    size_t rid, ibegin, iend, i;

    /* clear inst with all missing */
    for (i = 0; i < num_feature; ++i) {
      inst[i].missing = -1;
    }

    for (rid = 0; rid < nrow; ++rid) {
      ibegin = row_ptr[rid];
      iend = row_ptr[rid + 1];
      /* Fill nonzeros */
      for (i = ibegin; i < iend; ++i) {
        inst[col_ind[i]].fvalue = val[i];
      }
      out_pred[rid] = predict(inst, 0);
      /* Drop nonzeros */
      for (i = ibegin; i < iend; ++i) {
        inst[col_ind[i]].missing = -1;
      }
      printf("pred[%zu] = %f\n", rid, out_pred[rid]);
    }
    free(inst);
    free(out_pred);
    return 0;
  }

Save the program as a .c file and put it in the same directory ``mymodel/``. To
link the program against the prediction library ``mymodel.so``, simply run

.. code-block:: bash

  gcc -o myprog myprog.c mymodel.so -I. -std=c99

As long as the program ``myprog`` is in the same directory of the prediction
library ``mymodel.so``, we'll be good to go.

A sample output:

.. code-block:: none

  pred[0] = 44.880001
  pred[1] = 44.880001
  pred[2] = 44.880001
  pred[3] = 42.670002
  pred[4] = 44.880001
