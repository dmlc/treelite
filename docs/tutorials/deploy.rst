Deploying models
================

After all the hard work you did to train your tree ensemble model, you now have
to **deploy** the model. Deployment refers to distributing your model to
other machines and devices so as to make predictions on them. To facilitate
the coming discussions, let us define a few terms.

* **Host machine** : the machine running treelite.
* **Target machine** : the machine on which predictions will be made. The host
  machine may or may not be identical to the target machine. In cases where
  it's infeasible to install treelite on the target machine, the host and
  target machines will be necessarily distinct.
* **Shared library** : a blob of executable subroutines that can be imported by
  other native applications. Shared libraries will often have file extensions
  .dll, .so, or .dylib. Going back to the particular context of tree deployment,
  treelite will produce a shared library containing the prediction subroutine
  (compiled to native machine code).
* **Runtime package** : a :doc:`tiny fraction<../treelite-runtime-api>` of the
  full treelite package, consisting of a few helper functions that lets you
  easily load shared libraries and make predictions. The runtime is good to
  have, but on systems lacking Python we can do without it.

.. _deploy_option1:

Option 1: Install treelite on the target machine
------------------------------------------------
If feasible, this option is probably the most convenient. On the target machine,
install treelite by running pip:

.. code-block:: bash

  pip3 install treelite --user

Once treelite is installed, it suffices to follow instructions in
:doc:`../quick_start`.

This option is available if the target machine satisfies the following
conditions:

* One of the following C compiler is available: gcc, clang, Microsoft Visual C++.
* Python is installed, with version 2.7 or >= 3.5.
* The following Python packages are available: :py:mod:`numpy`, :py:mod:`scipy`,
  :py:mod:`pip`

In addition, if you are using operating systems other than Windows, Mac OS X,
and Linux, you would need to
:ref:`compile treelite from the source <install-source>`. To do this, you'll
need git and CMake (>= 3.1).

**[Under construction]**

.. _deploy_option2:

Option 2: Deploy prediction code with the runtime package
---------------------------------------------------------

**[Under construction]**

.. _deploy_option3:

Option 3: Deploy prediciton code only
-------------------------------------

**[Under construction]**
