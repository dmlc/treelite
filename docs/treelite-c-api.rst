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

.. doxygengroup:: opaque_handles
   :project: treelite
   :content-only:

