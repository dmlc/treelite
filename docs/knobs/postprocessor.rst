List of postprocessor functions
===============================
When predicting with tree ensemble models, we sum the margin scores from individual trees and apply a postprocessor
function to transform the sum into a final prediction. This function is also known as the link function.

Currently, Treelite supports the following postprocessor functions.

Element-wise postprocessor functions
------------------------------------
* ``identity``: The identity function. Do not apply any transformation to the margin score vector.
* ``signed_square``: Apply the function ``f(x) = sign(x) * (x**2)`` element-wise to the margin score vector.
* ``hinge``: Apply the function ``f(x) = (1 if x > 0 else 0)`` element-wise to the margin score vector.
* ``sigmoid``: Apply the sigmoid function ``f(x) = 1/(1+exp(-sigmoid_alpha * x))`` element-wise to the margin score
  vector, to transform margin scores into probability scores in the range ``[0, 1]``. The ``sigmoid_alpha`` parameter
  can be configured by the user.
* ``exponential``: Apply the exponential function (``exp``) element-wise to the margin score vector.
* ``exponential_standard_ratio``: Apply the function ``f(x) = exp2(-x / ratio_c)`` element-wise to the margin score
  vector. The ``ratio_c`` parameter can be configured by the user.
* ``logarithm_one_plus_exp``: Apply the function ``f(x) = log(1 + exp(x))`` element-wise to the margin score vector.

Row-wise postprocessor functions
--------------------------------
* ``identity_multiclass``:  The identity function. Do not apply any transformation to the margin score vector.
* ``softmax``: Use the softmax function ``f(x) = exp(x) / sum(exp(x))`` to the margin score vector, to transform the
  margin scores into probability scores in the range ``[0, 1]``. Adding up the transformed scores for all classes
  will yield 1.
* ``multiclass_ova``: Apply the sigmoid function ``f(x) = 1/(1+exp(-sigmoid_alpha * x))`` element-wise to the margin
  scores. The ``sigmoid_alpha`` parameter can be configured by the user.
