#-----------------------------------------------------
#  treelite: the configuration compile script
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of treelite.
#  First copy this file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries in the copied file and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#----------------------------------------------------

# choice of compiler, by default use system preference.
# export CC = gcc
# export CXX = g++
# export MPICXX = mpicxx

# the additional link flags you want to add
ADD_LDFLAGS =

# the additional compile flags you want to add
ADD_CFLAGS =

# Whether enable openmp support, needed for multi-threading.
USE_OPENMP = 1

# whether use HDFS support during compile
USE_HDFS = 0

# whether use AWS S3 support during compile
USE_S3 = 0

# whether use Azure blob support during compile
USE_AZURE = 0

# whether to test with coverage measurement or not. (only used for `make cover`)
# measured with gcov and html report generated with lcov if it is installed.
# this disables optimization to ensure coverage information is correct
TEST_COVER = 0

# path to gtest library (only used when $BUILD_TEST=1)
# there should be an include path in $GTEST_PATH/include and library in $GTEST_PATH/lib
GTEST_PATH =
