Treelite Release Notes 
======================

## 0.21 (03/08/2018)
New features:
* Now categorical features with more than 64 categories are supported.
* It is now possible to specify variants of GCC and Clang for `export_lib`;
  `gcc-5`, `gcc-7`, `arm-linux-gnueabi-gcc`, and so forth.

Bug fixes:
* Fix segmentation fault when `parallel_comp` is set to a high value

## 0.2 (02/12/2018)
A few minor fixes:
* Disable Link-Time Optimization (LTO) by default, to decrease compilation time
* Use /bin/sh when the environment variable SHELL is not set
* Enable relative path in predictor constructor
* Increase precision for floating-point values

## 0.1rc1 (11/15/2017)

* Initial release
