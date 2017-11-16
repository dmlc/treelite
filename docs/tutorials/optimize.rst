Optimizing prediction subroutine
================================

Treelite offers system-level optimizations to boost prediction performance.
Notice that the model information is wholly preserved; the optimizations only
affect the manner at which prediction is performed.

.. contents:: Contents
  :local:
  :backlinks: none
  :depth: 2

Annotate conditional branches
-----------------------------

This optimization analyzes and annotates every threshold conditions in the test
nodes to improve performance.

How to use
~~~~~~~~~~
The first step is to generate the **branch annotation record** for your ensemble
model. Make sure to have your training data ready.

.. code-block:: python

  # model = your ensemble model (object of type treelite.Model)
  # dmat = training data (object of type treelite.DMatrix)

  # Create annotator object
  annotator = treelite.Annotator()
  # Annotate branches by iterating over the training data
  annotator.annotate_branch(model=model, dmat=dmat, verbose=True)
  # Save the branch annotation record as a JSON file
  annotator.save(path='mymodel-annotation.json')

To utilize the branch annotation record, supply the compiler parameter
``annotate_in`` when exporting the model:

.. code-block:: python
  :emphasize-lines: 3, 8, 12

  # Export a source directory
  model.compile(dirpath='./mymodel', verbose=True,
                params={'quantize': 1})

  # Export a source directory, packaged in a zip archive
  model.export_srcpkg(platform='unix', toolchain='gcc', pkgpath='./mymodel.zip',
                      libname='mymodel.so', verbose=True,
                      params={'quantize': 1})

  # Export a shared library
  model.export_lib(toolchain='gcc', libpath='./mymodel.so', verbose=True,
                   params={'quantize': 1})    

Technical details
~~~~~~~~~~~~~~~~~

Rationale
^^^^^^^^^
Modern CPUs heavily rely on a technique known as
`branch prediction <https://en.wikipedia.org/wiki/Branch_predictor>`_, in which
they "guess" the result of the conditional expression in each ``if``-``else``
branch ahead of time. Given a program

.. code-block:: c

  if ( [conditional expression] ) {
    foo();
  } else {
    bar();
  }

the CPU will pre-fetch the instructions for the function ``foo()`` if the given
condition is likely to be true. On the other hand, if the condition is likely
to be false, the CPU will pre-fetch the instructions for the function ``bar()``.
It suffices to say that correctly prediction conditional branches has
great impact on performance. Each time the CPU predicts a branch correctly, it
can keep the instructions it had pre-fetched earlier. Each time the CPU fails to
predict, it must throw away the pre-fetched instructions and fetch anew another
set of instructions. If you'd like to learn more about the importance of branch
prediction, read
`this excellent introductory article
from Stanford\
<https://cs.stanford.edu/people/eroberts/courses/soco/projects/risc/pipelining/index.html>`_.

The prediction subroutine for a decision tree ensemble is problematic, as it
is replete with conditional branches that will have to be guessed well:

.. code-block:: c

  /* A slice of prediction subroutine */
  float predict_margin(const float* data) {
    float sum = 0.0f;
    if (!(data[0].missing != -1) || data[0].fvalue <= 9.5) {
      if (!(data[0].missing != -1) || data[0].fvalue <= 3.5) {
        if (!(data[10].missing != -1) || data[10].fvalue <= 0.74185) {
          if (!(data[0].missing != -1) || data[0].fvalue <= 1.5) {
            if (!(data[2].missing != -1) || data[2].fvalue <= 2.08671) {
              if ( (data[4].missing != -1) && data[4].fvalue <= 2.02632) {
                if (!(data[3].missing != -1) || data[3].fvalue <= 0.763339) {
                  sum += (float)0.00758165;
                } else {
                  sum += (float)0.0060202;
                }
              } else {
                if ( (data[1].missing != -1) && data[1].fvalue <= 0.0397456) {
                  sum += (float)0.00415399;
                } else {
                  sum += (float)0.00821985;
                }
              }
  /* and so forth... */

In fact, each threshold condition in the test nodes will need to be predicted.
While CPUs lack adequate information to make good guesses on these conditions, 
we can help by providing that information.

Mechanism for supplying the C compiler with branch information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We predict the likelihood of each condition by counting the number of data
points from the training data that satisfy that condition. See the diagram below
for an illustration.

.. plot::
  :nofigs:

  from graphviz import Source
  source = r"""
    digraph branch_annotation {
      graph [fontname = "helvetica"];
      node [fontname = "helvetica"];
      edge [fontname = "helvetica"];
      0  [label="Feature 4224 < 2.0 ?", shape=box];
      1  [label="Leaf: -0.0198"];
      2  [label="Feature 13 < 0.848 ?", shape=box];
      5  [label="Feature 12 < 0.878 ?", shape=box];
      6  [label="Feature 5 < -0.384 ?", shape=box];
      11 [label="         ⋮", margin=-0.5, shape=none, fontsize=20];
      12 [label="Feature 8 < -0.231?", shape=box];
      13 [label="Leaf: -0.0194"];
      14 [label="Leaf: -0.0196"];
      25 [label="Leaf: -0.0194"];
      26 [label="Feature 1693 < 2 ?", shape=box];
      53 [label="Leaf: 0"];
      54 [label="Leaf: -0.0196"];
      0  -> 1  [labeldistance=2.0, labelangle=45,
                headlabel=<Yes: <FONT COLOR="RED">3,092,211</FONT>                 >];
      0  -> 2  [labeldistance=2.0, labelangle=-45,
                headlabel=<                           No/Missing: <FONT COLOR="RED">3,902,211</FONT>>];
      2  -> 5  [labeldistance=2.0, labelangle=45,
                headlabel=<Yes: <FONT COLOR="RED">3,342,535</FONT>                 >];
      2  -> 6  [labeldistance=2.0, labelangle=-45,
                headlabel=<                           No/Missing: <FONT COLOR="RED">583,254</FONT>>];
      5  -> 11 [labeldistance=2.0, labelangle=45,
                headlabel=<Yes: <FONT COLOR="RED">2,878,952</FONT>                 >];
      5  -> 12 [labeldistance=2.0, labelangle=-45,
                headlabel=<               No/Missing:<BR/>          <FONT COLOR="RED">445,583</FONT>>];
      6  -> 13 [labeldistance=2.0, labelangle=45,
                headlabel=<Yes:  <BR/><FONT COLOR="RED">266,188</FONT>        >];
      6  -> 14 [labeldistance=2.0, labelangle=-45,
                headlabel=<         No/Missing:<BR/>               <FONT COLOR="RED">317,066</FONT>>];
      12 -> 25 [labeldistance=2.0, labelangle=45,
                headlabel=<Yes: <FONT COLOR="RED">257,828</FONT>                 >];
      12 -> 26 [labeldistance=2.0, labelangle=-45,
                headlabel=<                           No/Missing: <FONT COLOR="RED">187,755</FONT>>];
      26 -> 53 [labeldistance=2.0, labelangle=45,
                headlabel=<Yes: <FONT COLOR="RED">4</FONT>      >];
      26 -> 54 [labeldistance=2.0, labelangle=-45,
                headlabel=<                           No/Missing: <FONT COLOR="RED">187,751</FONT>>];
    }
  """
  Source(source, format='png').render('../_static/branch_annotation', view=False)
  Source(source, format='svg').render('../_static/branch_annotation', view=False)

.. raw:: html

  <p>
  <img src="../_static/branch_annotation.svg"
       onerror="this.src='../_static/branch_annotation.png'; this.onerror=null;">
  </p>

If a condition is true at least 50% of the time (over the training data), the
condition is labeled as "expected to be true":

.. code-block:: c

  /* expected to be true */
  if ( __builtin_expect( [condition], 1 ) ) {
    ...
  } else {
    ...
  }

On the other hand, if a condition is false at least 50% of the time, the
condition is labeled as "expected to be false":

.. code-block:: c

  /* expected to be false */
  if ( __builtin_expect( [condition], 0 ) ) {
    ...
  } else {
    ...
  }

.. note:: On the expression ``__builtin_expect``

  The ``__builtin_expect`` expression is a compiler intrinsic to supply the C
  compiler with branch prediction information. Both
  `gcc <https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fexpect>`_
  and `clang <https://llvm.org/docs/BranchWeightMetadata.html#built-in-expect-instructions>`_
  support it. Unfortunately, Microsoft Visual C++ does not. To take advantage
  of branch annotation, make sure to use gcc or clang on the target machine.       

Use integer thresholds for conditions
--------------------------------------

This optimization replaces all thresholds in the test nodes with integers so
that each threshold condition performs integer comparison instead of the usual
floating-point comparison. The thresholds are said to be **quantized** into
integer indices.

BEFORE:

.. code-block:: c

  if (data[3].fvalue < 1.5) {  /* floating-point comparison */
    ...
  }

AFTER:

.. code-block:: c

  if (data[3].qvalue < 3) {     /* integer comparison */
    ...
  }

How to use
~~~~~~~~~~
Simply add the compiler parameter ``quantize=1`` when exporting the model:

.. code-block:: python
  :emphasize-lines: 3, 8, 12

  # Export a source directory
  model.compile(dirpath='./mymodel', verbose=True,
                params={'quantize': 1})

  # Export a source directory, packaged in a zip archive
  model.export_srcpkg(platform='unix', toolchain='gcc', pkgpath='./mymodel.zip',
                      libname='mymodel.so', verbose=True,
                      params={'quantize': 1})

  # Export a shared library
  model.export_lib(toolchain='gcc', libpath='./mymodel.so', verbose=True,
                   params={'quantize': 1})    

Technical details
~~~~~~~~~~~~~~~~~

Rationale
^^^^^^^^^
On some platforms such as x86-64, replacing floating-point thresholds with
integers helps improve performance by 1) **reducing executable code size** and 2)
**improving data locality**. This is so because on these platforms, integer
constants can be embedded as part of the comparison instruction, whereas
floating-point constants cannot.

Let's look at x86-64 platform. The integer comparison

.. code-block:: c

  a <= 4

produces one assembly instruction:

.. code-block:: nasm

  cmpl    $4, 8(%rsp)       ;    8(%rsp) contains the variable a

Since the integer constant ``4`` got embedded into the comparison instruction
`cmpl <http://x86.renejeschke.de/html/file_module_x86_id_35.html>`_, we only
had to fetch the variable ``a`` from memory.

On the other hand, the floating-point comparison

.. code-block:: c

  b < 1.2f

produces two assembly instructions:

.. code-block:: nasm

  movss   250(%rip), %xmm0  ;  250(%rip) contains the constant 1.2f
  ucomiss  12(%rsp), %xmm0  ;   12(%rsp) contains the variable b

Notice that the floating-point constant ``1.2f`` did not get embedded into
the comparison instruction
`ucomiss <http://x86.renejeschke.de/html/file_module_x86_id_317.html>`_. The
constant had to be fetched (with
`movss <http://x86.renejeschke.de/html/file_module_x86_id_205.html>`_) into the
register ``xmm0`` before the comparsion could take place. To summarize,

* a floating-point comparison takes twice as many instructions as an integer
  comparsion, increasing the executable code size;
* a floating-point comparison involves an extra fetch instruction (``movss``),
  potentially causing a
  `cache miss <https://en.wikipedia.org/wiki/CPU_cache#Cache_miss>`_.

**Caveats**. As we'll see in the next section, using integer thresholds will
add overhead costs at prediction time. You should ensure that the benefits of
integer comparisons outweights the overhead costs.

Mechanism for mapping features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When ``quantize`` option is enabled, treelite will collect all thresholds
occuring in the tree ensemble model. For each feature, one list will be
generated that lists the thresholds in ascending order:

.. code-block:: none

  /* example of how per-feature threshold list may look like */

  Feature 0:  [1.5, 6.5, 12.5]
  Feature 3:  [0.15, 0.35, 1.5]
  Feature 6:  [7, 9, 10, 135]

Using these lists, we may convert any data point into integer indices via
simple look-ups. For feature 0 in the example above, values will be mapped
to integer indices as follows:

.. code-block:: none

  Let x be the value of feature 0.

  Assign -1 if          x  <  1.5
  Assign  0 if          x ==  1.5
  Assign  1 if   1.5  < x  <  6.5
  Assign  2 if          x ==  6.5
  Assign  3 if   6.5  < x  < 12.5
  Assign  4 if          x == 12.5
  Assign  5 if          x  > 12.5

Let's look at a specific example of how a floating-point vector gets translated
into a vector of integer indices:

.. code-block:: none
  
  feature id   0     1        2      3      4        5      6
              [7, missing, missing, 0.2, missing, missing, 20 ]
           => [3, missing, missing,   1, missing, missing,  5 ]

Since the prediction subroutine still needs to accept floating-point features,
the features will be internally converted before actual prediction. If the
prediction subroutine looked like below without ``quantize`` option,

.. code-block:: c

  float predict_margin(const Entry* data) {
    /* ... Run through the trees to compute the leaf output score ... */

    return score;
  }

it will now have an extra step of mapping the incoming data vector into integers:

.. code-block:: c
  :emphasize-lines: 2

  float predict_margin(const Entry* data) {
    /* ... Quantize feature values in data into integer indices   ... */

    /* ... Run through the trees to compute the leaf output score ... */
    return score;
  }