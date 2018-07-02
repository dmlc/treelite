import sys
import os
try:
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../runtime/native/python/'))
  from treelite_runtime import *
except ModuleNotFoundError:
  sys.path.insert(0, os.path.dirname(__file__))
  from treelite_runtime import *
