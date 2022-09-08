Notes on Serialization
======================

Treelite model objects can be serialized into two ways:

* `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_, to enable
  zero-copy serialization in the Python programming environment. When pickling a Python
  object containing a Treelite model object, we can convert the Treelite model into
  a byte sequence without physically making copies in memory.
* Files. Tree models can be converted into Treelite checkpoint files that can be later
  read back.
  
We make certain guarantees about compatiblity of serialization. It is possible to
exchange serialized tree models between two different Treelite versions, as follows:

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

+----------------------+--------------+--------------+--------------------+---------------+
|                      | To: ``=2.4`` | To: ``=3.0`` | To: ``>=3.1,<4.0`` | To: ``>=4.0`` |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``<2.4``       | |cross|      | |cross|      | |cross|            | |cross|       |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``=2.4``       | |tick|       | |tick|       | |tick|             | |cross|       |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``=3.0``       | |cross|      | |tick|       | |tick|             | |tick|        |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``>=3.1,<4.0`` | |cross|      | |tick|       | |tick|             | |tick|        |
+----------------------+--------------+--------------+--------------------+---------------+
| From: ``>=4.0``      | |cross|      | |cross|      | |cross|            | |tick|        |
+----------------------+--------------+--------------+--------------------+---------------+
