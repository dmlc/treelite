Installation
============

You may choose one of two methods to install Treelite on your system:

* :ref:`install-pip`
* :ref:`install-source`

.. _install-pip:

Download binary releases from PyPI (Recommended)
------------------------------------------------
This is probably the most convenient method. Simply type

.. code-block:: console

  python3 -m pip install --user treelite treelite_runtime

to install the Treelite package. The command will locate the binary release that is compatible with
your current platform. Check the installation by running

.. code-block:: python

  import treelite
  import treelite_runtime

in an interactive Python session. This method is available for only Windows, Mac OS X, and Linux.
For other operating systems, see the next section.

.. note:: Installing OpenMP runtime on Mac OSX
  
  Treelite requires the presence of OpenMP runtime. To install OpenMP runtime on a Mac OSX system,
  run the following command:

  .. code-block:: bash

    brew install libomp

.. _install-source:

Compile Treelite from the source
--------------------------------
Installation consists of two steps:

1. Build the shared libraries from C++ code (See the note below for the list.)
2. Install the Python package.

.. note:: List of libraries created

   There will be two libraries created: the main library, for producing optimized prediction
   subroutines; and the runtime library, for deploying these subroutines in the wild for actual
   prediction tasks.

   ================== ===================== =============================
   Operating System   Main library          Runtime library
   ================== ===================== =============================
   Windows            ``treelite.dll``      ``treelite_runtime.dll``
   Mac OS X           ``libtreelite.dylib`` ``libtreelite_runtime.dylib``
   Linux / other UNIX ``libtreelite.so``    ``libtreelite_runtime.so``
   ================== ===================== =============================

To get started, clone Treelite repo from GitHub.

.. code-block:: bash

  git clone https://github.com/dmlc/treelite.git
  cd treelite

The next step is to build the shared libraries.

1-1. Compiling shared libraries on Linux and Mac OS X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here, we use CMake to generate a Makefile:

.. code-block:: bash

  mkdir build
  cd build
  cmake ..

Once CMake finished running, simply invoke GNU Make to obtain the shared
libraries.

.. code-block:: bash

  make

The compiled libraries will be under the ``build/`` directory.

.. note:: Compiling Treelite with multithreading on Mac OS X

  Treelite requires the presence of OpenMP runtime. To install OpenMP runtime on a Mac OSX system,
  run the following command:

  .. code-block:: bash

    brew install libomp

1-2. Compiling shared libraries on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can use CMake to generate a Visual Studio project. The following snippet assumes that Visual
Studio 2017 is installed. Adjust the version depending on the copy that's installed on your system.

.. code-block:: dosbatch

  mkdir build
  cd build
  cmake .. -G"Visual Studio 15 2017 Win64"

.. note:: Visual Studio 2017 or newer is required

  Ensure that you have Visual Studio version 2017 or newer.

Once CMake finished running, open the generated solution file (``treelite.sln``) in Visual Studio.
From the top menu, select **Build > Build Solution**.

2. Installing Python package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Python package is located at the ``python`` subdirectory. There are several
ways to install the package:

**1. Install system-wide, which requires root permission**

.. code-block:: bash

  # Install treelite
  cd python
  sudo python3 setup.py install
  # Install treelite_runtime
  cd ../runtime/python
  sudo python3 setup.py install

You will need Python `setuptools <https://pypi.python.org/pypi/setuptools>`_
module for this to work. It is often part of the core Python installation.
Should it be necessary, the package can be installed using ``pip``:

.. code-block:: bash

  pip install -U pip setuptools

**2. Install for only current user**

This is useful if you do not have the administrative rights.

.. code-block:: bash

  # Install treelite
  cd python
  python3 setup.py install --user
  # Install treelite_runtime
  cd ../runtime/python
  python3 setup.py install --user

.. note:: Recompiling Treelite

  Every time the C++ portion of Treelite gets re-compiled, the Python
  package must be re-installed for the new library to take effect.

**3. Set the environment variable PYTHONPATH to locate Treelite package**

Only set the environment variable ``PYTHONPATH`` to tell Python where to find
the Treelite package. This is useful for developers, as any changes made
to C++ code will be immediately visible to Python side without re-running ``setup.py``.

.. code-block:: bash

  export PYTHONPATH=/path/to/treelite/python:/path/to/treelite/runtime/python
  python3          # enter interactive session

