============
Installation
============

You may choose one of two methods to install Treelite on your system:

.. contents::
  :local:
  :depth: 1

Download binary releases from PyPI (Recommended)
================================================
This is probably the most convenient method. Simply type

.. code-block:: console

  pip install treelite

to install the Treelite package. The command will locate the binary release that is compatible with
your current platform. Check the installation by running

.. code-block:: python

  import treelite

in an interactive Python session. This method is available for only Windows, MacOS, and Linux.
For other operating systems, see the next section.

.. note:: Windows users need to install Visual C++ Redistributable

  Treelite requires DLLs from `Visual C++ Redistributable
  <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`_
  in order to function, so make sure to install it. Exception: If
  you have Visual Studio installed, you already have access to
  necessary libraries and thus don't need to install Visual C++
  Redistributable.

.. note:: Installing OpenMP runtime on MacOS
  
  Treelite requires the presence of OpenMP runtime. To install OpenMP runtime on a MacOS system,
  run the following command:

  .. code-block:: bash

    brew install libomp


Download binary releases from Conda
===================================
Treelite is also available on Conda.

.. code-block:: console

  conda install -c conda-forge treelite

to install the Treelite package. See https://anaconda.org/conda-forge/treelite to check the
available platforms.

.. _install-source:

Compile Treelite from the source
================================
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
   MacOS              ``libtreelite.dylib`` ``libtreelite_runtime.dylib``
   Linux / other UNIX ``libtreelite.so``    ``libtreelite_runtime.so``
   ================== ===================== =============================

To get started, clone Treelite repo from GitHub.

.. code-block:: bash

  git clone https://github.com/dmlc/treelite.git
  cd treelite

The next step is to build the shared libraries.

1-1. Compiling shared libraries on Linux and MacOS
--------------------------------------------------
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

.. note:: Compiling Treelite with multithreading on MacOS

  Treelite requires the presence of OpenMP runtime. To install OpenMP runtime on a Mac OSX system,
  run the following command:

  .. code-block:: bash

    brew install libomp

1-2. Compiling shared libraries on Windows
------------------------------------------
We can use CMake to generate a Visual Studio project. The following snippet assumes that Visual
Studio 2022 is installed. Adjust the version depending on the copy that's installed on your system.

.. code-block:: dosbatch

  mkdir build
  cd build
  cmake .. -G"Visual Studio 17 2022" -A x64

.. note:: Visual Studio 2019 or newer is required

  Treelite uses the C++17 standard. Ensure that you have Visual Studio version 2019 or newer.

Once CMake finished running, open the generated solution file (``treelite.sln``) in Visual Studio.
From the top menu, select **Build > Build Solution**.

2. Installing Python package
----------------------------
The Python package is located at the ``python`` subdirectory. Run Pip to install the Python
package. The Python package will re-use the native library built in Step 1.

.. code-block:: bash

  cd python
  pip install .  # will re-use libtreelite.so
