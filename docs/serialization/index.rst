Notes on Serialization
======================

Treelite model objects can be serialized into three ways:

* Byte sequence. A tree model can be serialized to a byte sequence in memory. Later,
  we can recover the same tree model by deserializing it from the byte sequence.
* Files. Tree models can be converted into Treelite checkpoint files that can be later
  read back.
* `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_, to enable
  zero-copy serialization in the Python programming environment. When pickling a Python
  object containing a Treelite model object, we can convert the Treelite model into
  a byte sequence without physically making copies in memory.

We make certain guarantees about compatiblity of serialization. It is possible to
exchange serialized tree models between two different Treelite versions, as follows:

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

+----------------------+--------------+--------------+--------------------+---------------+
|                      | To: ``=3.9`` | To: ``=4.0`` | To: ``>=4.1,<5.0`` | To: ``>=5.0`` |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``=3.9``       | |tick|       | |tick|       | |tick|             | |cross|       |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``=4.0``       | |cross|      | |tick|       | |tick|             | |tick|        |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``>=4.1,<5.0`` | |cross|      | |tick|       | |tick|             | |tick|        |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``>=5.0``      | |cross|      | |cross|      | |cross|            | |tick|        |
+----------------------+--------------+--------------+--------------------+---------------+

.. toctree::
 :maxdepth: 1

 v3
