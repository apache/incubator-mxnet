# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Namespace for operators used in Gluon dispatched by F=symbol."""

from __future__ import absolute_import
from . import _symbol
from . import _op as _mx_sym_np
from . import _internal as _npi

__all__ = ['norm', 'svd', 'cholesky']


def norm(x, ord=None, axis=None, keepdims=False):
    r"""Matrix or vector norm.

    This function can only support Frobenius norm for now.
    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    Parameters
    ----------
    x : ndarray
        Input array.
    ord : {'fro'}, optional
        Order of the norm.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None, the norm of the whole ndarray is
        returned.

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
    """
    if ord is not None and ord != 'fro':
        raise ValueError('only support Frobenius norm for now, received ord={}'.format(str(ord)))
    if isinstance(axis, tuple) and len(axis) > 2:
        raise ValueError('Improper number of dimensions to norm')
    # TODO(junwu): When ord = 'fro', axis = None, and x.ndim > 2, raise exception
    return _symbol.sqrt(_mx_sym_np.sum(x * x, axis=axis, keepdims=keepdims))


def svd(a):
    r"""
    Singular Value Decomposition.

    When `a` is a 2D array, it is factorized as ``ut @ np.diag(s) @ v``,
    where `ut` and `v` are 2D orthonormal arrays and `s` is a 1D
    array of `a`'s singular values. When `a` is higher-dimensional, SVD is
    applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) _Symbol
        A real array with ``a.ndim >= 2`` and ``M <= N``.

    Returns
    -------
    ut: (..., M, M) _Symbol
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    s : (..., M) _Symbol
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    v : (..., M, N) _Symbol
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.

    Notes
    -----

    The decomposition is performed using LAPACK routine ``_gesvd``.

    SVD is usually described for the factorization of a 2D matrix :math:`A`.
    The higher-dimensional case will be discussed below. In the 2D case, SVD is
    written as :math:`A = U^T S V`, where :math:`A = a`, :math:`U^T = ut`,
    :math:`S= \mathtt{np.diag}(s)` and :math:`V = v`. The 1D array `s`
    contains the singular values of `a` and `ut` and `v` are orthonormal. The rows
    of `v` are the eigenvectors of :math:`A^T A` and the columns of `ut` are
    the eigenvectors of :math:`A A^T`. In both cases the corresponding
    (possibly non-zero) eigenvalues are given by ``s**2``.

    The sign of rows of `u` and `v` are determined as described in
    `Auto-Differentiating Linear Algebra <https://arxiv.org/pdf/1710.08717.pdf>`_.

    If `a` has more than two dimensions, then broadcasting rules apply.
    This means that SVD is working in "stacked" mode: it iterates over
    all indices of the first ``a.ndim - 2`` dimensions and for each
    combination SVD is applied to the last two indices. The matrix `a`
    can be reconstructed from the decomposition with either
    ``(ut * s[..., None, :]) @ v`` or
    ``ut @ (s[..., None] * v)``. (The ``@`` operator denotes batch matrix multiplication)

    This function differs from the original `numpy.linalg.svd
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html>`_ in
    the following way(s):
     - The sign of rows of `u` and `v` may differ.
     - Does not support complex input.
    """
    return _npi.svd(a)


def cholesky(a):
    return _npi.cholesky(a)
