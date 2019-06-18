# coding: utf-8
"""branch annotator module"""

import ctypes
from .common.util import c_str, TreeliteError
from .core import _LIB, DMatrix, _check_call
from .frontend import Model

class Annotator():
  """
  Branch annotator class: annotate branches in a given model using frequency
  patterns in the training data
  """
  def __init__(self):
    self.handle = None

  def annotate_branch(self, model, dmat, nthread=None, verbose=False):
    """
    Annotate branches in a given model using frequency patterns in the
    training data. Each node gets the count of the instances that belong to it.
    Any prior annotation information stored in the annotator will be replaced
    with the new annotation returned by this method.

    Parameters
    ----------
    model : object of type :py:class:`Model`
        decision tree ensemble model
    dmat : object of type :py:class:`DMatrix`
        data matrix representing the training data
    nthread : :py:class:`int <python:int>`, optional
        number of threads to use while annotating. If missing, use all physical
        cores in the system.
    verbose : :py:class:`bool <python:bool>`, optional
        whether to produce extra messages
    """
    if not isinstance(model, Model):
      raise ValueError('model must be of Model type')
    if not isinstance(dmat, DMatrix):
      raise TreeliteError('dmat must be of type DMatrix')
    nthread = nthread if nthread is not None else 0
    tmp = ctypes.c_void_p()
    _check_call(_LIB.TreeliteAnnotateBranch(model.handle, dmat.handle,
                                            ctypes.c_int(nthread),
                                            ctypes.c_int(1 if verbose else 0),
                                            ctypes.byref(tmp)))
    if self.handle is None:
      self.handle = tmp
    else:
      # replace handle
      _check_call(_LIB.TreeliteAnnotationFree(self.handle))
      self.handle = tmp

  def save(self, path):
    """
    Save branch annotation infromation as a JSON file.

    Parameters
    ----------
    path : :py:class:`str <python:str>`
        location of saved JSON file
    """
    if self.handle is None:
      raise TreeliteError('Annotator is currently empty; either load from '+\
                          'an annotation file (.json) or call the '+\
                          'annotate_branch() method')
    _check_call(_LIB.TreeliteAnnotationSave(self.handle, c_str(path)))

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreeliteAnnotationFree(self.handle))
      self.handle = None

__all__ = ['Annotator']
