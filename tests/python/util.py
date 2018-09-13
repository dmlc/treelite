"""Utility functions for tests"""
import os
from sys import platform as _platform
import numpy as np
from nose.tools import nottest
import treelite
import treelite.runtime
from treelite.contrib import _libext

def load_txt(filename):
  """Get 1D array from text file"""
  if filename is None:
    return None
  content = []
  with open(filename, 'r') as f:
    for line in f:
      content.append(float(line))
  return np.array(content)

def os_compatible_toolchains():
  if _platform == 'darwin':
    toolchains = ['gcc-7', 'clang']
  elif _platform == 'win32':
    toolchains = ['msvc']
  else:
    toolchains = ['gcc', 'clang']
  return toolchains

def os_platform():
  if _platform == 'darwin':
    return 'osx'
  elif _platform == 'win32' or _platform == 'cygwin':
    return 'windows'
  else:
    return 'unix'

def libname(fmt):
  return fmt.format(_libext())

def make_annotation(model, dtrain_path, annotation_path):
  dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))
  dtrain_path = os.path.join(dpath, dtrain_path)
  dtrain = treelite.DMatrix(dtrain_path)
  annotator = treelite.Annotator()
  annotator.annotate_branch(model=model, dmat=dtrain, verbose=True)
  annotator.save(path=annotation_path)

@nottest
def run_pipeline_test(model, dtest_path, libname_fmt,
                      expected_prob_path, expected_margin_path,
                      multiclass, use_annotation, use_quantize,
                      use_parallel_comp):
  dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))
  dtest_path = os.path.join(dpath, dtest_path)
  libpath = libname(libname_fmt)
  dtest = treelite.DMatrix(dtest_path)
  batch = treelite.runtime.Batch.from_csr(dtest)

  expected_prob_path = os.path.join(dpath, expected_prob_path) \
                       if expected_prob_path is not None else None
  expected_margin_path = os.path.join(dpath, expected_margin_path)
  expected_prob = load_txt(expected_prob_path)
  expected_margin = load_txt(expected_margin_path)
  if multiclass:
    nrow = dtest.shape[0]
    expected_prob = expected_prob.reshape((nrow, -1)) \
                    if expected_prob is not None else None
    expected_margin = expected_margin.reshape((nrow, -1))
  params = {}
  if use_annotation is not None:
    params['annotate_in'] = use_annotation
  if use_quantize:
    params['quantize'] = 1
  if use_parallel_comp is not None:
    params['parallel_comp'] = use_parallel_comp

  for toolchain in os_compatible_toolchains():
    model.export_lib(toolchain=toolchain, libpath=libpath,
                     params=params, verbose=True)
    predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
    out_prob = predictor.predict(batch)
    if expected_prob is not None:
      assert np.allclose(out_prob, expected_prob, atol=1e-11, rtol=1e-6)
    out_margin = predictor.predict(batch, pred_margin=True)
    assert np.allclose(out_margin, expected_margin, atol=1e-11, rtol=1e-6)
