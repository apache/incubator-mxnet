# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments, wrong-import-order
# pylint: disable=consider-using-enumerate
"""Symbolic Executor component of MXNet."""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import mx_uint, NDArrayHandle, ExecutorHandle
from .base import check_call, c_array, py_str
from .ndarray import NDArray
from . import ndarray as nd
from .context import cpu
import logging

class Executor(object):
    """ Executor is the actual executing object of MXNet."""
    def __init__(self, handle, symbol):
        """Constructor, used Symbol.bind and Symbol.simple_bind instead.

        Parameters
        ----------
        handle: ExecutorHandle
            ExecutorHandle generated by calling Bind

        See Also
        --------
        Symbol.bind : to create executor
        """
        if not isinstance(handle, ExecutorHandle):
            raise TypeError("Handle type error")
        self.handle = handle
        self.arg_arrays = []
        self.grad_arrays = []
        self.aux_arrays = []
        self.outputs = self._get_outputs()
        self._symbol = symbol
        self._arg_dict = None
        self._grad_dict = None
        self._aux_dict = None
        self._monitor_callback = None

    def __del__(self):
        check_call(_LIB.MXExecutorFree(self.handle))

    @staticmethod
    def _get_dict(names, ndarrays):
        """Get the dictionary given name and ndarray pairs."""
        nset = set()
        for nm in names:
            if nm in nset:
                raise ValueError('Duplicate names detected, %s' % str(names))
            nset.add(nm)
        return dict(zip(names, ndarrays))

    def _get_outputs(self):
        """list all the output ndarray

        Returns
        -------
        A list of ndarray binded to the heads of executor.
        """
        out_size = mx_uint()
        handles = ctypes.POINTER(NDArrayHandle)()
        check_call(_LIB.MXExecutorOutputs(self.handle,
                                          ctypes.byref(out_size), ctypes.byref(handles)))
        return [NDArray(NDArrayHandle(handles[i])) for i in range(out_size.value)]

    def forward(self, is_train=False, **kwargs):
        """Calculate the outputs specified by the binded symbol.

        Parameters
        ----------
        is_train: bool, optional
            whether this forward is for evaluation purpose.

        **kwargs
            Additional specification of input arguments.

        Examples
        --------
        >>> # doing forward by specifying data
        >>> texec.forward(is_train=True, data=mydata)
        >>> # doing forward by not specifying things, but copy to the executor before hand
        >>> mydata.copyto(texec.arg_dict['data'])
        >>> texec.forward(is_train=True)
        """
        if len(kwargs) != 0:
            arg_dict = self.arg_dict
            for name, array in kwargs.items():
                if not isinstance(array, NDArray):
                    raise ValueError('only accept keyword argument of NDArrays')
                if name not in arg_dict:
                    raise TypeError('Unknown argument %s' % name)
                array.copyto(arg_dict[name])

        check_call(_LIB.MXExecutorForward(
            self.handle,
            ctypes.c_int(int(is_train))))

    def backward(self, out_grads=None):
        """Do backward pass to get the gradient of arguments.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        if out_grads is None:
            out_grads = []
        elif isinstance(out_grads, NDArray):
            out_grads = [out_grads]

        for obj in out_grads:
            if not isinstance(obj, NDArray):
                raise TypeError("inputs must be NDArray")
        ndarray = c_array(NDArrayHandle, [item.handle for item in out_grads])
        check_call(_LIB.MXExecutorBackward(
            self.handle,
            mx_uint(len(out_grads)),
            ndarray))

    def set_monitor_callback(self, callback):
        """Install callback.

        Parameters
        ----------
        callback : function
            Takes a string and an NDArrayHandle.
        """
        cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, NDArrayHandle)
        self._monitor_callback = cb_type(callback)
        check_call(_LIB.MXExecutorSetMonitorCallback(
            self.handle,
            self._monitor_callback))

    @property
    def arg_dict(self):
        """Get dictionary representation of argument arrrays.

        Returns
        -------
        arg_dict : dict of str to NDArray
            The dictionary that maps name of arguments to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the arguments.
        """
        if self._arg_dict is None:
            self._arg_dict = Executor._get_dict(
                self._symbol.list_arguments(), self.arg_arrays)
        return self._arg_dict

    @property
    def aux_dict(self):
        """Get dictionary representation of auxiliary states arrays.

        Returns
        -------
        aux_dict : dict of str to NDArray
            The dictionary that maps name of auxiliary states to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the auxiliary states.
        """
        if self._aux_dict is None:
            self._aux_dict = Executor._get_dict(
                self._symbol.list_auxiliary_states(), self.aux_arrays)
        return self._aux_dict

    def copy_params_from(self, arg_params, aux_params=None, allow_extra_params=False):
        """Copy parameters from arg_params, aux_params into executor's internal array.

        Parameters
        ----------
        arg_params : dict of str to NDArray
            Parameters, dict of name to NDArray of arguments

        aux_params : dict of str to NDArray, optional
            Parameters, dict of name to NDArray of auxiliary states.

        allow_extra_params : boolean, optional
            Whether allow extra parameters that are not needed by symbol
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.

        Raises
        ------
        ValueError
            If there is additional parameters in the dict but allow_extra_params=False
        """
        for name, array in arg_params.items():
            if name in self.arg_dict:
                array.copyto(self.arg_dict[name])
            else:
                if not allow_extra_params:
                    raise ValueError('Find name \"%s\" that is not in the arguments' % name)
        if aux_params is None:
            aux_params = {}
        for name, array in aux_params.items():
            if name in self.aux_dict:
                array.copyto(self.aux_dict[name])
            else:
                if not allow_extra_params:
                    raise ValueError('Find name %s that is not in the auxiliary states' % name)

    def debug_str(self):
        """Get a debug string about internal execution plan.

        Returns
        -------
        debug_str : string
            Debug string of the executor.
        """
        debug_str = ctypes.c_char_p()
        check_call(_LIB.MXExecutorPrint(
            self.handle, ctypes.byref(debug_str)))
        return py_str(debug_str.value)

def _split_input_slice(batch_size, work_load_list):
    """Get input slice from the input shape.
    Parameters
    ----------
    batch_size : int
        The number of samples in a mini-batch.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ctx
    Returns
    -------
    slices : list of slice
        The split slices to get a specific slice.
    Raises
    ------
    ValueError
        If there are two many splits such that some slice can be empty.
    """
    total_work_load = sum(work_load_list)
    batch_num_list = [round(work_load * batch_size / total_work_load)
                      for work_load in work_load_list]
    batch_num_sum = sum(batch_num_list)
    if batch_num_sum < batch_size:
        batch_num_list[-1] += batch_size - batch_num_sum
    slices = []
    end = 0
    for batch_num in batch_num_list:
        begin = int(min((end, batch_size)))
        end = int(min((begin + batch_num, batch_size)))
        if begin >= end:
            raise ValueError('Too many slices such that some splits are empty')
        slices.append(slice(begin, end))
    return slices

def _check_arguments(symbol):
    """Check the argument names of symbol.
    This function checks the duplication of arguments in Symbol.
    The check is done for feedforward net for now.
    Parameters
    ----------
    symbol : Symbol
        The network configuration
    """
    arg_set = set()
    arg_names = symbol.list_arguments()
    for name in arg_names:
        if name in arg_set:
            raise ValueError(('Find duplicated argument name \"%s\", ' +
                              'please make the weight name non-duplicated(using name arguments), ' +
                              'arguments are %s') % (name, str(arg_names)))
        arg_set.add(name)

    aux_set = set()
    aux_names = symbol.list_auxiliary_states()
    for name in aux_names:
        if name in aux_set:
            raise ValueError(
                ('Find duplicated auxiliary param name \"%s\", ' +
                 'please make the weight name non-duplicated(using name arguments), ' +
                 'arguments are %s, auxiliary params are %s'
                ) % (name, str(arg_names), str(aux_names)))
        aux_set.add(name)

def _load_general(data, targets):
    """Load a list of arrays into a list of arrays specified by slices"""
    for d_src, d_targets in zip(data, targets):
        if isinstance(d_targets, nd.NDArray):
            d_src.copyto(d_targets)
        else:
            for slice_idx, d_dst in d_targets:
                d_src[slice_idx].copyto(d_dst)

def _load_data(batch, targets):
    """Load data into sliced arrays"""
    _load_general(batch.data, targets)

def _load_label(batch, targets):
    """Load label into sliced arrays"""
    _load_general(batch.label, targets)

class DataParallelExecutorManager(object):
    """ Helper class to manage multiple executors for data parallelism.
    Parameters
    ----------
    symbol : Symbol
        output symbol
    ctx : list of Context
        devices to run on
    param_names: list of str
        Name of all trainable parameters of the network.
    arg_names: list of str
        Name of all arguments of the network.
    aux_names: list of str
        Name of all auxiliary states of the network.
    train_data : DataIter
        Training data iterator.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ctx
    logger : logging logger
        When not specified, default logger will be used.
    """
    def __init__(self, symbol, ctx, train_data,
                 param_names, arg_names, aux_names,
                 work_load_list=None, logger=None):
        if logger is None:
            logger = logging
        # preparation
        num_device = len(ctx)
        logger.info('Start training with %s', str(ctx))

        # make sure the architecture is valid
        _check_arguments(symbol)

        if work_load_list is None:
            work_load_list = [1] * num_device
        assert isinstance(work_load_list, list) and len(work_load_list) == num_device, \
            "Invalid settings for work load. "

        slices = _split_input_slice(train_data.batch_size, work_load_list)
        self.slices = slices

        self.train_execs = []
        for i in range(len(ctx)):
            data_shapes = {k: tuple([slices[i].stop-slices[i].start] + list(v[1:]))
                           for k, v in train_data.provide_data}
            train_exec = symbol.simple_bind(ctx[i], 'write', **data_shapes)
            self.train_execs.append(train_exec)

        # data structure
        self.data_names = [x[0] for x in train_data.provide_data]
        self.label_names = [x[0] for x in train_data.provide_label]
        self.aux_names = aux_names

        self.data_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                            for name in self.data_names]
        self.label_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                             for name in self.label_names]

        self.param_idx = [i for i in range(len(arg_names)) if arg_names[i] in param_names]
        self.param_names = [arg_names[i] for i in self.param_idx]
        self.param_arrays = [[e.arg_arrays[i] for e in self.train_execs]
                             for i in self.param_idx]
        self.grad_arrays = [[e.grad_arrays[i] for e in self.train_execs]
                            for i in self.param_idx]

        self.aux_arrays = [[e.aux_arrays[i] for e in self.train_execs]
                           for i in range(len(aux_names))]

        batch_size = train_data.batch_size

        output_shapes = [tuple([batch_size]+list(x.shape[1:])) for x in self.train_execs[0].outputs]
        self.cpu_output_arrays = [nd.zeros(s) for s in output_shapes]

    def install_monitor(self, monitor):
        """ Install monitor on all executors """
        for train_exec in self.train_execs:
            monitor.install(train_exec)

    def set_params(self, arg_params, aux_params):
        """ set parameter and aux values
        Parameters
        ----------
        arg_params : list of NDArray
            source parameter arrays
        aux_params : list of NDArray
            source aux arrays
        """

        for texec in self.train_execs:
            texec.copy_params_from(arg_params, aux_params)

    def copy_to(self, arg_params, aux_params):
        """ Copy data from each executor to `arg_params` and `aux_params`
        Parameters
        ----------
        arg_params : list of NDArray
            target parameter arrays
        aux_params : list of NDArray
            target aux arrays
        Notes
        -----
        - This function will inplace update the NDArrays in arg_params and aux_params.
        """
        for name, block in zip(self.param_names, self.param_arrays):
            weight = sum(w.copyto(cpu()) for w in block) / len(block)
            weight.copyto(arg_params[name])
        for name, block in zip(self.aux_names, self.aux_arrays):
            weight = sum(w.copyto(cpu()) for w in block) / len(block)
            weight.copyto(aux_params[name])

    def load_data_batch(self, data_batch):
        """ load data and labels into arrays """
        _load_data(data_batch, self.data_arrays)
        _load_label(data_batch, self.label_arrays)

    def forward(self, is_train=False):
        """ Perform a forward pass on each executor """
        for texec, islice in zip(self.train_execs, self.slices):
            texec.forward(is_train=is_train)
            for cpu_out, dev_out in zip(self.cpu_output_arrays, texec.outputs):
                dev_out.copyto(cpu_out[islice])

    def backward(self):
        """ Perform a backward pass on each executor """
        for texec in self.train_execs:
            texec.backward()
