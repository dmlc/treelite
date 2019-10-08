# coding: utf-8
"""Frontend collection for treelite"""
from __future__ import absolute_import as _abs
import ctypes
import collections
import shutil
import os
from tempfile import TemporaryDirectory
from .common.compat import STRING_TYPES
from .common.util import c_str, TreeliteError
from .core import _LIB, c_array, _check_call
from .contrib import create_shared, generate_makefile, _toolchain_exist_check

def _isascii(string):
  """Tests if a given string is pure ASCII; works for both Python 2 and 3"""
  try:
    return len(string) == len(string.encode())
  except UnicodeDecodeError:
    return False
  except UnicodeEncodeError:
    return False

class Model():
  """
  Decision tree ensemble model

  Parameters
  ----------
  handle : :py:class:`ctypes.c_void_p <python:ctypes.c_void_p>`, optional
      Initial value of model handle
  """
  def __init__(self, handle=None):
    if handle is None:
      self.handle = None
    else:
      if not isinstance(handle, ctypes.c_void_p):
        raise ValueError('Model handle must be of type ctypes.c_void_p')
      self.handle = handle

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreeliteFreeModel(self.handle))
      self.handle = None

  def set_tree_limit(self, n):
    """Set first n trees to be kept, the remaining ones will be dropped"""
    if self.handle is None:
      raise AttributeError('Model not loaded yet')
    _check_call(_LIB.TreeliteSetTreeLimit(self.handle, ctypes.c_size_t(n)))

  @property
  def num_tree(self):
    """Number of decision trees in the model"""
    if self.handle is None:
      raise AttributeError('Model not loaded yet')
    out = ctypes.c_size_t()
    _check_call(_LIB.TreeliteQueryNumTree(self.handle, ctypes.byref(out)))
    return out.value

  @property
  def num_feature(self):
    """Number of features used in the model"""
    if self.handle is None:
      raise AttributeError('Model not loaded yet')
    out = ctypes.c_size_t()
    _check_call(_LIB.TreeliteQueryNumFeature(self.handle, ctypes.byref(out)))
    return out.value

  @property
  def num_output_group(self):
    """Number of output groups of the model"""
    if self.handle is None:
      raise AttributeError('Model not loaded yet')
    out = ctypes.c_size_t()
    _check_call(_LIB.TreeliteQueryNumOutputGroups(self.handle, ctypes.byref(out)))
    return out.value

  # pylint: disable=R0913
  def export_lib(self, toolchain, libpath, params=None, compiler='ast_native',
                 verbose=False, nthread=None, options=None):
    """
    Convenience function: Generate prediction code and immediately turn it
    into a dynamic shared library. A temporary directory will be created to
    hold the source files.

    Parameters
    ----------
    toolchain : :py:class:`str <python:str>`
      which toolchain to use. You may choose one of 'msvc', 'clang', and 'gcc'.
      You may also specify a specific variation of clang or gcc (e.g. 'gcc-7')
    libpath : :py:class:`str <python:str>`
        location to save the generated dynamic shared library
    params : :py:class:`dict <python:dict>`, optional
        parameters to be passed to the compiler. See
        :py:doc:`this page <knobs/compiler_param>` for the list of compiler
        parameters.
    compiler : :py:class:`str <python:str>`, optional
        name of compiler to use in C code generation
    verbose : :py:class:`bool <python:bool>`, optional
        whether to produce extra messages
    nthread : :py:class:`int <python:int>`, optional
        number of threads to use in creating the shared library.
        Defaults to the number of cores in the system.
    options : :py:class:`list <python:list>` of :py:class:`str <python:str>`, \
              optional
        Additional options to pass to toolchain

    Example
    -------
    The one-line command

    .. code-block:: python

       model.export_lib(toolchain='msvc', libpath='./mymodel.dll',
                        params={}, verbose=True)

    is equivalent to the following sequence of commands:

    .. code-block:: python

       model.compile(dirpath='/temporary/directory', params={}, verbose=True)
       treelite.create_shared(toolchain='msvc', dirpath='/temporary/directory',
                     verbose=True)
       # move the library out of the temporary directory
       shutil.move('/temporary/directory/mymodel.dll', './mymodel.dll')
    """
    _toolchain_exist_check(toolchain)
    with TemporaryDirectory(dir=os.path.dirname(libpath)) as temp_dir:
      self.compile(temp_dir, params, compiler, verbose)
      temp_libpath = create_shared(toolchain, temp_dir, nthread,
                                   verbose, options)
      if os.path.exists(libpath) and os.path.isfile(libpath):
        os.remove(libpath)
      shutil.move(temp_libpath, libpath)

  def export_srcpkg(self, platform, toolchain, pkgpath, libname, params=None,
                    compiler='ast_native', verbose=False, options=None):
    """
    Convenience function: Generate prediction code and create a zipped source
    package for deployment. The resulting zip file will also contain a Makefile.

    Parameters
    ----------
    platform : :py:class:`str <python:str>`
      name of the operating system on which the headers and sources shall be
      compiled. Must be one of the following: 'windows' (Microsoft Windows),
      'osx' (Mac OS X), 'unix' (Linux and other UNIX-like systems)
    toolchain : :py:class:`str <python:str>`
      which toolchain to use. You may choose one of 'msvc', 'clang', and 'gcc'.
      You may also specify a specific variation of clang or gcc (e.g. 'gcc-7')
    pkgpath : :py:class:`str <python:str>`
        location to save the zipped source package
    libname : :py:class:`str <python:str>`
        name of model shared library to be built
    params : :py:class:`dict <python:dict>`, optional
        parameters to be passed to the compiler. See
        :py:doc:`this page <knobs/compiler_param>` for the list of compiler
        parameters.
    compiler : :py:class:`str <python:str>`, optional
        name of compiler to use in C code generation
    verbose : :py:class:`bool <python:bool>`, optional
        whether to produce extra messages
    nthread : :py:class:`int <python:int>`, optional
        number of threads to use in creating the shared library.
        Defaults to the number of cores in the system.
    options : :py:class:`list <python:list>` of :py:class:`str <python:str>`, \
              optional
        Additional options to pass to toolchain

    Example
    -------
    The one-line command

    .. code-block:: python

       model.export_srcpkg(platform='unix', toolchain='gcc',
                           pkgpath='./mymodel_pkg.zip', libname='mymodel.so',
                           params={}, verbose=True)

    is equivalent to the following sequence of commands:

    .. code-block:: python

       model.compile(dirpath='/temporary/directory/mymodel',
                     params={}, verbose=True)
       generate_makefile(dirpath='/temporary/directory/mymodel',
                         platform='unix', toolchain='gcc')
       # zip the directory containing C code and Makefile
       shutil.make_archive(base_name=pkgpath, format='zip',
                           root_dir='/temporary/directory',
                           base_dir='mymodel/')
    """
    # check for file extension
    fileext = os.path.splitext(pkgpath)[1]
    if fileext != '.zip':
      raise ValueError('Source package file should have .zip extension')
    libname = os.path.basename(libname)
    _toolchain_exist_check(toolchain)

    with TemporaryDirectory() as temp_dir:
      target = os.path.splitext(libname)[0]
      # create a child directory to get desired name for target
      dirpath = os.path.join(temp_dir, target)
      os.makedirs(dirpath)
      if params is None:
        params = {}
      params['native_lib_name'] = target
      self.compile(dirpath, params, compiler, verbose)
      generate_makefile(dirpath, platform, toolchain, options)
      shutil.make_archive(base_name=os.path.splitext(pkgpath)[0],
                          format='zip',
                          root_dir=temp_dir,
                          base_dir='{}/'.format(target))

  def compile(self, dirpath, params=None, compiler='ast_native', verbose=False):
    """
    Generate prediction code from a tree ensemble model. The code will be C99
    compliant. One header file (.h) will be generated, along with one or more
    source files (.c). Use :py:meth:`create_shared` method to package
    prediction code as a dynamic shared library (.so/.dll/.dylib).

    Parameters
    ----------
    dirpath : :py:class:`str <python:str>`
        directory to store header and source files
    params : :py:class:`dict <python:dict>`, optional
        parameters for compiler. See
        :py:doc:`this page <knobs/compiler_param>` for the list of compiler
        parameters.
    compiler : :py:class:`str <python:str>`, optional
        name of compiler to use
    verbose : :py:class:`bool <python:bool>`, optional
        Whether to print extra messages during compilation

    Example
    -------
    The following populates the directory ``./model`` with source and header
    files:

    .. code-block:: python

       model.compile(dirpath='./my/model', params={}, verbose=True)

    If parallel compilation is enabled (parameter ``parallel_comp``), the files
    are in the form of ``./my/model/header.h``, ``./my/model/main.c``,
    ``./my/model/tu0.c``, ``./my/model/tu1.c`` and so forth, depending on
    the value of ``parallel_comp``. Otherwise, there will be exactly two files:
    ``./model/header.h``, ``./my/model/main.c``
    """
    compiler_handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteCompilerCreate(c_str(compiler),
                                            ctypes.byref(compiler_handle)))
    _params = dict(params) if isinstance(params, list) else params
    self._set_compiler_param(compiler_handle, _params or {})
    _check_call(_LIB.TreeliteCompilerGenerateCode(
        compiler_handle,
        self.handle,
        ctypes.c_int(1 if verbose else 0),
        c_str(dirpath)))
    _check_call(_LIB.TreeliteCompilerFree(compiler_handle))

  def export_protobuf(self, filename):
    """
    Export a tree ensemble model as a Protocol Buffers format. Protocol Buffers
    (google/protobuf) is a language- and platform-neutral mechanism for
    serializing structured data. See src/tree.proto for format spec.

    Parameters
    ----------
    filename : :py:class:`str <python:str>`
        path to save Protocol Buffers output

    Example
    -------
    .. code-block:: python

       model.export_protobuf('./my.buffer')
    """
    _check_call(_LIB.TreeliteExportProtobufModel(c_str(filename), self.handle))

  @staticmethod
  def _set_compiler_param(compiler_handle, params, value=None):
    """
    Set parameter(s) for compiler

    Parameters
    ----------
    params: :py:class:`dict <python:dict>` / :py:class:`list <python:list>` / \
            :py:class:`str <python:str>`
        list of key-alue pairs, dict or simply string key
    compiler_handle: object of type `ctypes.c_void_p`
        handle to compiler
    value: optional
        value of the specified parameter, when params is a single string
    """
    if isinstance(params, collections.Mapping):
      params = params.items()
    elif isinstance(params, STRING_TYPES) and value is not None:
      params = [(params, value)]
    for key, val in params:
      _check_call(_LIB.TreeliteCompilerSetParam(compiler_handle, c_str(key),
                                                c_str(str(val))))
  @classmethod
  def from_xgboost(cls, booster):
    """
    Load a tree ensemble model from an XGBoost Booster object

    Parameters
    ----------
    booster : object of type :py:class:`xgboost.Booster`
        Python handle to XGBoost model

    Returns
    -------
    model : :py:class:`Model` object
        loaded model

    Example
    -------

    .. code-block:: python
       :emphasize-lines: 2

       bst = xgboost.train(params, dtrain, 10, [(dtrain, 'train')])
       xgb_model = Model.from_xgboost(bst)
    """
    handle = ctypes.c_void_p()
    # attempt to load xgboost
    try:
      import xgboost
    except ImportError:
      raise TreeliteError('xgboost module must be installed to read from '+\
                          '`xgboost.Booster` object')
    if not isinstance(booster, xgboost.Booster):
      raise ValueError('booster must be of type `xgboost.Booster`')
    buffer = booster.save_raw()
    ptr = (ctypes.c_char * len(buffer)).from_buffer(buffer)
    length = ctypes.c_size_t(len(buffer))
    _check_call(_LIB.TreeliteLoadXGBoostModelFromMemoryBuffer(
        ptr,
        length,
        ctypes.byref(handle)))
    return Model(handle)

  @classmethod
  def load(cls, filename, model_format):
    """
    Load a tree ensemble model from a file

    Parameters
    ----------
    filename : :py:class:`str <python:str>`
        path to model file
    model_format : :py:class:`str <python:str>`
        model file format. Must be one or 'xgboost', 'lightgbm', 'protobuf'

    Returns
    -------
    model : :py:class:`Model` object
        loaded model

    Example
    -------

    .. code-block:: python

       xgb_model = Model.load('xgboost_model.model', 'xgboost')
    """
    handle = ctypes.c_void_p()
    if not _isascii(model_format):
      raise ValueError('model_format parameter must be an ASCII string')
    model_format = model_format.lower()
    if model_format == 'lightgbm':
      _check_call(_LIB.TreeliteLoadLightGBMModel(c_str(filename),
                                                 ctypes.byref(handle)))
    elif model_format == 'xgboost':
      _check_call(_LIB.TreeliteLoadXGBoostModel(c_str(filename),
                                                ctypes.byref(handle)))
    elif model_format == 'protobuf':
      _check_call(_LIB.TreeliteLoadProtobufModel(c_str(filename),
                                                 ctypes.byref(handle)))
    else:
      raise ValueError('Unknown model_format: must be one of ' \
                        + '{lightgbm, xgboost, protobuf}')
    return Model(handle)

class ModelBuilder():
  """
  Builder class for tree ensemble model: provides tools to iteratively build
  an ensemble of decision trees

  Parameters
  ----------
  num_feature : :py:class:`int <python:int>`
      number of features used in model being built. We assume that all
      feature indices are between ``0`` and (``num_feature - 1``)
  num_output_group : :py:class:`int <python:int>`, optional
      number of output groups; ``>1`` indicates multiclass classification
  random_forest : :py:class:`bool <python:bool>`, optional
      whether the model is a random forest; ``True`` indicates a random forest
      and ``False`` indicates gradient boosted trees
  **kwargs
      model parameters, to be used to specify the resulting model. Refer to
      :py:doc:`this page <knobs/model_param>` for the full list of model
      parameters.
  """
  class Node():
    """Handle to a node in a tree"""
    def __init__(self):
      self.empty = True

    def __repr__(self):
      return '<treelite.ModelBuilder.Node object>'

    def set_root(self):
      """
      Set the node as the root
      """
      try:
        _check_call(_LIB.TreeliteTreeBuilderSetRootNode(
            self.tree.handle,
            ctypes.c_int(self.node_key)))
      except AttributeError:
        raise TreeliteError('This node has never been inserted into a tree; '\
                           + 'a node must be inserted before it can be a root')

    def set_leaf_node(self, leaf_value):
      """
      Set the node as a leaf node

      Parameters
      ----------
      leaf_value : :py:class:`float <python:float>` / \
                   :py:class:`list <python:list>` of \
                   :py:class:`float <python:float>`
          Usually a single leaf value (weight) of the leaf node. For multiclass
          random forest classifier, leaf_value should be a list of leaf weights.
      """

      if not self.empty:
        try:
          node_key = self.node_key
        except AttributeError:
          node_key = '_'
        raise ValueError(
            'Cannot modify a non-empty node. ' + \
            'If you meant to change type of node {}, '.format(node_key) + \
            'delete it first and then add an empty node with ' + \
            'the same key.')

      # check if leaf_value is a list-like object
      try:
        _ = iter(leaf_value)
        is_list = True
      except TypeError:
        is_list = False

      try:
        if is_list:
          leaf_value = [float(i) for i in leaf_value]
        else:
          leaf_value = float(leaf_value)
      except TypeError:
        raise TreeliteError('leaf_value parameter should be either a ' + \
                            'single float or a list of floats')

      try:
        if is_list:
          _check_call(_LIB.TreeliteTreeBuilderSetLeafVectorNode(
              self.tree.handle,
              ctypes.c_int(self.node_key),
              c_array(ctypes.c_double, leaf_value),
              ctypes.c_size_t(len(leaf_value))))
        else:
          _check_call(_LIB.TreeliteTreeBuilderSetLeafNode(
              self.tree.handle,
              ctypes.c_int(self.node_key),
              ctypes.c_double(leaf_value)))
        self.empty = False
      except AttributeError:
        raise TreeliteError('This node has never been inserted into a tree; '\
                      + 'a node must be inserted before it can be a leaf node')

    # pylint: disable=R0913
    def set_numerical_test_node(self, feature_id, opname, threshold,
                                default_left, left_child_key, right_child_key):
      """
      Set the node as a test node with numerical split. The test is in the form
      ``[feature value] OP [threshold]``. Depending on the result of the test,
      either left or right child would be taken.

      Parameters
      ----------
      feature_id : :py:class:`int <python:int>`
          feature index
      opname : :py:class:`str <python:str>`
          binary operator to use in the test
      threshold : :py:class:`float <python:float>`
          threshold value
      default_left : :py:class:`bool <python:bool>`
          default direction for missing values
          (``True`` for left; ``False`` for right)
      left_child_key : :py:class:`int <python:int>`
          unique integer key to identify the left child node
      right_child_key : :py:class:`int <python:int>`
          unique integer key to identify the right child node
      """
      if not self.empty:
        try:
          node_key = self.node_key
        except AttributeError:
          node_key = '_'
        raise ValueError(
            'Cannot modify a non-empty node. ' + \
            'If you meant to change type of node {}, '.format(node_key) + \
            'delete it first and then add an empty node with ' + \
            'the same key.')
      try:
        # automatically create child nodes that don't exist yet
        if left_child_key not in self.tree:
          self.tree[left_child_key] = ModelBuilder.Node()
        if right_child_key not in self.tree:
          self.tree[right_child_key] = ModelBuilder.Node()
        _check_call(_LIB.TreeliteTreeBuilderSetNumericalTestNode(
            self.tree.handle,
            ctypes.c_int(self.node_key),
            ctypes.c_uint(feature_id), c_str(opname),
            ctypes.c_double(threshold),
            ctypes.c_int(1 if default_left else 0),
            ctypes.c_int(left_child_key),
            ctypes.c_int(right_child_key)))
        self.empty = False
      except AttributeError:
        raise TreeliteError('This node has never been inserted into a tree; '\
                      + 'a node must be inserted before it can be a test node')

    # pylint: disable=R0913
    def set_categorical_test_node(self, feature_id, left_categories,
                                  default_left, left_child_key,
                                  right_child_key):
      """
      Set the node as a test node with categorical split. A list defines all
      categories that would be classified as the left side. Categories are
      integers ranging from ``0`` to ``n-1``, where ``n`` is the number of
      categories in that particular feature.

      Parameters
      ----------
      feature_id : :py:class:`int <python:int>`
          feature index
      left_categories : :py:class:`list <python:list>` of \
                        :py:class:`int <python:int>`
          list of categories belonging to the left child.
      default_left : :py:class:`bool <python:bool>`
          default direction for missing values
          (``True`` for left; ``False`` for right)
      left_child_key : :py:class:`int <python:int>`
          unique integer key to identify the left child node
      right_child_key : :py:class:`int <python:int>`
          unique integer key to identify the right child node
      """
      if not self.empty:
        try:
          node_key = self.node_key
        except AttributeError:
          node_key = '_'
        raise ValueError(
            'Cannot modify a non-empty node. ' + \
            'If you meant to change type of node {}, '.format(node_key) + \
            'delete it first and then add an empty node with ' + \
            'the same key.')
      try:
        # automatically create child nodes that don't exist yet
        if left_child_key not in self.tree:
          self.tree[left_child_key] = ModelBuilder.Node()
        if right_child_key not in self.tree:
          self.tree[right_child_key] = ModelBuilder.Node()
        _check_call(_LIB.TreeliteTreeBuilderSetCategoricalTestNode(
            self.tree.handle,
            ctypes.c_int(self.node_key),
            ctypes.c_uint(feature_id),
            c_array(ctypes.c_uint, left_categories),
            ctypes.c_size_t(len(left_categories)),
            ctypes.c_int(1 if default_left else 0),
            ctypes.c_int(left_child_key),
            ctypes.c_int(right_child_key)))
        self.empty = False
      except AttributeError:
        raise TreeliteError('This node has never been inserted into a tree; '\
                      + 'a node must be inserted before it can be a test node')

  class Tree():
    """Handle to a decision tree in a tree ensemble Builder"""
    def __init__(self):
      self.handle = ctypes.c_void_p()
      _check_call(_LIB.TreeliteCreateTreeBuilder(ctypes.byref(self.handle)))
      self.nodes = {}

    def __del__(self):
      if self.handle is not None:
        if not hasattr(self, 'ensemble'):
          # need a separate deletion if tree is not part of an ensemble
          _check_call(_LIB.TreeliteDeleteTreeBuilder(self.handle))
        self.handle = None

    ### Implement dict semantics whenever applicable
    def items(self):              # pylint: disable=C0111
      return self.nodes.items()

    def keys(self):               # pylint: disable=C0111
      return self.nodes.keys()

    def values(self):             # pylint: disable=C0111
      return self.nodes.values()

    def __len__(self):
      return len(self.nodes)

    def __getitem__(self, key):
      if key not in self.nodes:
        # implicitly create a new node
        self.__setitem__(key, ModelBuilder.Node())
      return self.nodes.__getitem__(key)

    def __setitem__(self, key, value):
      if not isinstance(value, ModelBuilder.Node):
        raise ValueError('Value must be of type ModelBuidler.Node')
      if key in self.nodes:
        raise KeyError('Nodes with duplicate keys are not allowed. ' + \
                       'If you meant to replace node {}, '.format(key) + \
                       'delete it first and then add an empty node with ' + \
                       'the same key.')
      if not value.empty:
        raise ValueError('Can only insert an empty node')
      _check_call(_LIB.TreeliteTreeBuilderCreateNode(self.handle,
                                                     ctypes.c_int(key)))
      self.nodes.__setitem__(key, value)
      value.node_key = key  # save node id for later
      value.tree = self

    def __delitem__(self, key):
      _check_call(_LIB.TreeliteTreeBuilderDeleteNode(
          self.handle, ctypes.c_int(key)))
      self.nodes.__delitem__(key)

    def __iter__(self):
      return self.nodes.__iter__()

    def __repr__(self):
      return '<treelite.ModelBuilder.Tree object containing {} nodes>\n'\
             .format(len(self.nodes))

  def __init__(self, num_feature, num_output_group=1,
               random_forest=False, **kwargs):
    if not isinstance(num_feature, int):
      raise ValueError('num_feature must be of int type')
    if num_feature <= 0:
      raise ValueError('num_feature must be strictly positive')
    if not isinstance(num_output_group, int):
      raise ValueError('num_output_group must be of int type')
    if num_output_group <= 0:
      raise ValueError('num_output_group must be strictly positive')
    self.handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteCreateModelBuilder(
        ctypes.c_int(num_feature),
        ctypes.c_int(num_output_group),
        ctypes.c_int(1 if random_forest else 0),
        ctypes.byref(self.handle)))
    self._set_param(kwargs)
    self.trees = []

  def insert(self, index, tree):
    """
    Insert a tree at specified location in the ensemble

    Parameters
    ----------
    index : :py:class:`int <python:int>`
        index of the element before which to insert the tree
    tree : :py:class:`.Tree` object
        tree to be inserted

    Example
    -------

    .. code-block:: python
       :emphasize-lines: 3

       builder = ModelBuilder(num_feature=4227)
       tree = ...               # build tree somehow
       builder.insert(0, tree)  # insert tree at index 0

    """
    if not isinstance(index, int):
      raise ValueError('index must be of int type')
    if index < 0 or index > len(self):
      raise ValueError('index out of bounds')
    if not isinstance(tree, ModelBuilder.Tree):
      raise ValueError('tree must be of type ModelBuilder.Tree')
    ret = _LIB.TreeliteModelBuilderInsertTree(self.handle,
                                              tree.handle,
                                              ctypes.c_int(index))
    _check_call(0 if ret == index else -1)
    if ret != index:
      raise ValueError('Somehow tree got inserted at wrong location')
    # delete the stale handle to the inserted tree and get a new one
    _check_call(_LIB.TreeliteDeleteTreeBuilder(tree.handle))
    _check_call(_LIB.TreeliteModelBuilderGetTree(self.handle,
                                                 ctypes.c_int(index),
                                                 ctypes.byref(tree.handle)))
    tree.ensemble = self
    self.trees.insert(index, tree)

  def append(self, tree):
    """
    Add a tree at the end of the ensemble

    Parameters
    ----------
    tree : :py:class:`.Tree` object
        tree to be added

    Example
    -------
    .. code-block:: python
       :emphasize-lines: 3

       builder = ModelBuilder(num_feature=4227)
       tree = ...               # build tree somehow
       builder.append(tree)     # add tree at the end of the ensemble
    """
    self.insert(len(self), tree)

  def commit(self):
    """
    Finalize the ensemble model

    Returns
    -------
    model : :py:class:`Model` object
        finished model

    Example
    -------
    .. code-block:: python
       :emphasize-lines: 6

       builder = ModelBuilder(num_feature=4227)
       for i in range(100):
         tree = ...                    # build tree somehow
         builder.append(tree)          # add one tree at a time

       model = builder.commit()        # now get a Model object
       model.compile(dirpath='test')   # compile model into C code
    """
    model_handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteModelBuilderCommitModel(
        self.handle,
        ctypes.byref(model_handle)))
    return Model(model_handle)

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreeliteDeleteModelBuilder(self.handle))
      self.handle = None

  ### Implement list semantics whenever applicable
  def __len__(self):
    return len(self.trees)

  def __getitem__(self, index):
    return self.trees.__getitem__(index)

  def __delitem__(self, index):
    _check_call(_LIB.TreeliteModelBuilderDeleteTree(self.handle,
                                                    ctypes.c_int(index)))
    self.trees[index].handle = None  # handle is already invalid
    self.trees.__delitem__(index)

  def __iter__(self):
    return self.trees.__iter__()

  def __reversed__(self):
    return self.trees.__reversed__()

  def __repr__(self):
    return '<treelite.ModelBuilder object storing {} decision trees>\n'\
           .format(len(self.trees))

  def _set_param(self, params, value=None):
    """
    Set parameter(s)

    Parameters
    ----------
    params: dict / list / string
        list of key-alue pairs, dict or simply string key
    value: optional
        value of the specified parameter, when params is a single string
    """
    if isinstance(params, collections.Mapping):
      params = params.items()
    elif isinstance(params, STRING_TYPES) and value is not None:
      params = [(params, value)]
    for key, val in params:
      if not isinstance(val, STRING_TYPES):
        val = str(val)
      _check_call(_LIB.TreeliteModelBuilderSetModelParam(self.handle,
                                                         c_str(key),
                                                         c_str(val)))

__all__ = ['Model', 'ModelBuilder']
