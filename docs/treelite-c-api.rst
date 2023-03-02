==============
Treelite C API
==============

Treelite exposes a set of C functions to enable interfacing with a variety of
languages. This page will be most useful for:

* those writing a new
  `language binding <https://en.wikipedia.org/wiki/Language_binding>`_ (glue
  code).
* those wanting to incorporate functions of Treelite into their own native
  libraries.

**We recommend the Python API for everyday uses.**

.. note:: Use of C and C++ in Treelite

  Core logic of Treelite are written in C++ to take advantage of higher
  abstractions. We provide C only interface here, as many more programming
  languages bind with C than with C++. See
  `this page <https://softwareengineering.stackexchange.com/q/281882>`_ for
  more details.

.. contents:: Contents
  :local:
  :backlinks: none

Data matrix interface
---------------------
Use the following functions to load and manipulate data from a variety of
sources.

.. doxygengroup:: dmatrix
   :project: treelite
   :content-only:

Branch annotator interface
--------------------------
Use the following functions to annotate branches in decision trees.

.. doxygengroup:: annotator
   :project: treelite
   :content-only:

Compiler interface
------------------
Use the following functions to produce optimize prediction subroutine (in C)
from a given decision tree ensemble.

.. doxygengroup:: compiler
   :project: treelite
   :content-only:

Model loader interface
----------------------
Use the following functions to load decision tree ensemble models from a file.
Treelite supports multiple model file formats.

.. doxygengroup:: model_loader
   :project: treelite
   :content-only:

Model builder interface
-----------------------
Use the following functions to incrementally build decisio n tree ensemble
models.

.. doxygengroup:: model_builder
   :project: treelite
   :content-only:

Predictor interface
-------------------
Use the following functions to load compiled prediction subroutines
from shared libraries and to make predictions.

.. doxygengroup:: predictor
   :project: treelite
   :content-only:

General Tree Inference Library (GTIL)
-------------------------------------

.. doxygengroup:: gtil
   :project: treelite
   :content-only:

Handle types
------------
Treelite uses C++ classes to define its internal data structures. In order to
pass C++ objects to C functions, *opaque handles* are used. Opaque handles
are ``void*`` pointers that store raw memory addresses.

.. doxygengroup:: Opaque_handles
   :project: treelite
   :content-only:

