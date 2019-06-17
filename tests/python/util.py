"""Utility functions for tests"""
import os
from sys import platform as _platform
import numpy as np
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
    toolchains = ['gcc']
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

def get_atol(atol=None):
  """Get default numerical threshold for regression test."""
  return 1e-8 if atol is None else atol

def get_rtol(rtol=None):
  """Get default numerical threshold for regression test."""
  return 1e-3 if rtol is None else rtol

def find_max_violation(a, b, rtol=None, atol=None):
  """Finds and returns the location of maximum violation."""
  rtol = get_rtol(rtol)
  atol = get_atol(atol)
  diff = np.abs(a-b)
  tol = atol + rtol * np.abs(b)
  violation = diff / (tol + 1e-20)
  loc = np.argmax(violation)
  idx = np.unravel_index(loc, violation.shape)
  return idx, np.max(violation)

def assert_almost_equal(a, b, rtol=None, atol=None, names=('a', 'b'), equal_nan=False):
  """Test that two numpy arrays are almost equal. Raise exception message if not.
  Parameters
  ----------
  a : np.ndarray
  b : np.ndarray
  threshold : None or float
      The checking threshold. Default threshold will be used if set to ``None``.
  """
  rtol = get_rtol(rtol)
  atol = get_atol(atol)
  if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan):
    return
  index, rel = find_max_violation(a, b, rtol, atol)
  np.set_printoptions(threshold=4, suppress=True)
  msg = np.testing.build_err_msg([a, b],
                          err_msg="Error %f exceeds tolerance rtol=%f, atol=%f. "
                                  " Location of maximum error:%s, a=%f, b=%f"
                          % (rel, rtol, atol, str(index), a[index], b[index]),
                          names=names)
  raise AssertionError(msg)

def run_pipeline_test(model, dtest_path, libname_fmt,
                      expected_prob_path, expected_margin_path,
                      multiclass, use_annotation, use_quantize,
                      use_parallel_comp, use_code_folding=None,
                      use_toolchains=None):
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
  if use_code_folding is not None:
    params['code_folding_req'] = use_code_folding

  if use_toolchains is None:
    toolchains = os_compatible_toolchains()
  else:
    gcc = os.environ.get('GCC_PATH', 'gcc')
    toolchains = [(gcc if x == 'gcc' else x) for x in use_toolchains]
  for toolchain in toolchains:
    model.export_lib(toolchain=toolchain, libpath=libpath,
                     params=params, verbose=True)
    predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
    out_prob = predictor.predict(batch)
    if expected_prob is not None:
      assert_almost_equal(out_prob, expected_prob)
    out_margin = predictor.predict(batch, pred_margin=True)
    assert_almost_equal(out_margin, expected_margin)
