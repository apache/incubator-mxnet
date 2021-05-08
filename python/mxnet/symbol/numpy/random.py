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

import numpy as np
from ...context import current_context
from ...util import is_np_default_dtype
from . import _internal as _npi


__all__ = ['randint', 'uniform', 'normal', 'multivariate_normal',
           'logistic', 'gumbel', 'rayleigh', 'f',
           'rand', 'shuffle', 'gamma', 'beta', 'chisquare', 'exponential', 'lognormal',
           'weibull', 'pareto', 'power', 'laplace', 'geometric']


def randint(low, high=None, size=None, dtype=None, ctx=None, out=None):
    r"""Return random integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. All dtypes are determined by their
        name, i.e., 'int64', 'int', etc, so byteorder is not available
        and a specific precision may have different C types depending
        on the platform. The default value is 'np.int'.
    ctx : Context, optional
        Device context of output. Default is current context.
    out : _Symbol, optional
        The output symbol (default is `None`).

    Returns
    -------
    out : _Symbol
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    Examples
    --------
    >>> np.random.randint(2, size=10)
    array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
    >>> np.random.randint(1, size=10)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Generate a 2 x 4 array of ints between 0 and 4, inclusive:

    >>> np.random.randint(5, size=(2, 4))
    array([[4, 0, 2, 1],
        [3, 2, 2, 0]])
    """
    if dtype is None:
        dtype = 'int'
    if ctx is None:
        ctx = current_context()
    if size is None:
        size = ()
    if high is None:
        high = low
        low = 0
    return _npi.random_randint(low, high, shape=size, dtype=dtype, ctx=ctx, out=out)


def rand(*size, **kwargs):
    r"""Random values in a given shape.

    Create an array of the given shape and populate it with random
    samples from a uniform distribution over [0, 1).
    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned array, should be all positive.
        If no argument is given a single Python float is returned.
    Returns
    -------
    out : _Symbol
       Random values.
    Examples
    --------
    >>> np.random.rand(3,2)
    array([[ 0.14022471,  0.96360618],  #random
           [ 0.37601032,  0.25528411],  #random
           [ 0.49313049,  0.94909878]]) #random
    """
    output_shape = ()
    for s in size:
        output_shape += (s,)
    return uniform(0, 1, size=output_shape, **kwargs)


def uniform(low=0.0, high=1.0, size=None, dtype=None, ctx=None, out=None):
    r"""Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float, _Symbol, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float, _Symbol, optional
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a scalar tensor containing a single value is returned if
        ``low`` and ``high`` are both scalars.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples.
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol
        Drawn samples from the parameterized uniform distribution.
    """
    from ._symbol import _Symbol as np_symbol
    input_type = (isinstance(low, np_symbol), isinstance(high, np_symbol))
    if ctx is None:
        ctx = current_context()
    if out is not None:
        size = out.shape
    if size == ():
        size = None
    if input_type == (True, True):
        return _npi.uniform(low, high, low=None, high=None, size=size,
                            ctx=ctx, dtype=dtype, out=out)
    elif input_type == (False, True):
        return _npi.uniform(high, low=low, high=None, size=size,
                            ctx=ctx, dtype=dtype, out=out)
    elif input_type == (True, False):
        return _npi.uniform(low, low=None, high=high, size=size,
                            ctx=ctx, dtype=dtype, out=out)
    else:
        return _npi.uniform(low=low, high=high, size=size,
                            ctx=ctx, dtype=dtype, out=out)


def normal(loc=0.0, scale=1.0, size=None, dtype=None, ctx=None, out=None):
    r"""Draw random samples from a normal (Gaussian) distribution.

    Samples are distributed according to a normal distribution parametrized
    by *loc* (mean) and *scale* (standard deviation).


    Parameters
    ----------
    loc : float, optional
        Mean (centre) of the distribution.
    scale : float, optional
        Standard deviation (spread or "width") of the distribution.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., `(m, n, k)`, then `m * n * k`
        samples are drawn. If size is `None` (default), a scalar tensor containing
        a single value is returned if loc and scale are both scalars.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples.
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol (symbol representing `mxnet.numpy.ndarray` in computational graphs)
        Drawn samples from the parameterized normal distribution.
    """
    from ._symbol import _Symbol as np_symbol
    input_type = (isinstance(loc, np_symbol), isinstance(scale, np_symbol))
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    if input_type == (True, True):
        return _npi.normal(loc, scale, loc=None, scale=None, size=size,
                           ctx=ctx, dtype=dtype, out=out)
    elif input_type == (False, True):
        return _npi.normal(scale, loc=loc, scale=None, size=size,
                           ctx=ctx, dtype=dtype, out=out)
    elif input_type == (True, False):
        return _npi.normal(loc, loc=None, scale=scale, size=size,
                           ctx=ctx, dtype=dtype, out=out)
    else:
        return _npi.normal(loc=loc, scale=scale, size=size,
                           ctx=ctx, dtype=dtype, out=out)


def lognormal(mean=0.0, sigma=1.0, size=None, dtype=None, ctx=None, out=None):
    r"""Draw samples from a log-normal distribution.

    Draw samples from a log-normal distribution with specified mean,
    standard deviation, and array shape.  Note that the mean and standard
    deviation are not the values for the distribution itself, but of the
    underlying normal distribution it is derived from.

    Parameters
    ----------
    mean : float, optional
        Mean value of the underlying normal distribution. Default is 0.
    sigma : float, optional
        Standard deviation of the underlying normal distribution. Must be
        non-negative. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``mean`` and ``sigma`` are both scalars.
        Otherwise, ``np.broadcast(mean, sigma).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol (symbol representing `mxnet.numpy.ndarray` in computational graphs)
        Drawn samples from the parameterized lognormal distribution.
    """
    from . import _symbol as _mx_np_symbol
    return _mx_np_symbol.exp(normal(loc=mean, scale=sigma, size=size, dtype=dtype, ctx=ctx, out=out))


def logistic(loc=0.0, scale=1.0, size=None, ctx=None, out=None):
    r"""Draw samples from a logistic distribution.

    Samples are drawn from a logistic distribution with specified
    parameters, loc (location or mean, also median), and scale (>0).

    Parameters
    ----------
    loc : float, optional
        Parameter of the distribution. Default is 0.
    scale : float, optional
        Parameter of the distribution. Must be non-negative.
        Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol (symbol representing `mxnet.numpy.ndarray` in computational graphs)
        Drawn samples from the parameterized logistic distribution.
    """
    from ._symbol import _Symbol as np_symbol
    input_type = (isinstance(loc, np_symbol), isinstance(scale, np_symbol))
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    if input_type == (True, True):
        return _npi.logistic(loc, scale, loc=None, scale=None, size=size,
                             ctx=ctx, out=out)
    elif input_type == (False, True):
        return _npi.logistic(scale, loc=loc, scale=None, size=size,
                             ctx=ctx, out=out)
    elif input_type == (True, False):
        return _npi.logistic(loc, loc=None, scale=scale, size=size,
                             ctx=ctx, out=out)
    else:
        return _npi.logistic(loc=loc, scale=scale, size=size,
                             ctx=ctx, out=out)


def gumbel(loc=0.0, scale=1.0, size=None, ctx=None, out=None):
    r"""Draw samples from a Gumbel distribution.

    Parameters
    ----------
    loc : float or array_like of floats, optional
        The location of the mode of the distribution. Default is 0.
    scale : float or array_like of floats, optional
        The scale parameter of the distribution. Default is 1. Must be non-
        negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol (symbol representing `mxnet.numpy.ndarray` in computational graphs)
        Drawn samples from the parameterized gumbel distribution.
    """
    from ._symbol import _Symbol as np_symbol
    input_type = (isinstance(loc, np_symbol), isinstance(scale, np_symbol))
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    if input_type == (True, True):
        return _npi.gumbel(loc, scale, loc=None, scale=None, size=size,
                           ctx=ctx, out=out)
    elif input_type == (False, True):
        return _npi.gumbel(scale, loc=loc, scale=None, size=size,
                           ctx=ctx, out=out)
    elif input_type == (True, False):
        return _npi.gumbel(loc, loc=None, scale=scale, size=size,
                           ctx=ctx, out=out)
    else:
        return _npi.gumbel(loc=loc, scale=scale, size=size,
                           ctx=ctx, out=out)


def choice(a, size=None, replace=True, p=None, ctx=None, out=None):
    r"""Generates a random sample from a given 1-D array

    Parameters
    -----------
    a : 1-D array-like or int
        If an ndarray, a random sample is generated from its elements.
        If an int, the random sample is generated as if a were np.arange(a)
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    --------
    samples : _Symbol
        The generated random samples

    Examples
    ---------
    Generate a uniform random sample from np.arange(5) of size 3:

    >>> np.random.choice(5, 3)
    array([0, 3, 4])
    >>> #This is equivalent to np.random.randint(0,5,3)

    Generate a non-uniform random sample from np.arange(5) of size 3:

    >>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
    array([3, 3, 0])

    Generate a uniform random sample from np.arange(5) of size 3 without
    replacement:

    >>> np.random.choice(5, 3, replace=False)
    array([3,1,0])
    >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]

    Generate a non-uniform random sample from np.arange(5) of size
    3 without replacement:

    >>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
    array([2, 3, 0])
    """
    from ._symbol import _Symbol as np_symbol
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    if isinstance(a, np_symbol):
        ctx = None
        if p is None:
            indices = _npi.choice(a, a=None, size=size,
                                  replace=replace, ctx=ctx, weighted=False)
            return _npi.take(a, indices)
        else:
            indices = _npi.choice(a, p, a=None, size=size,
                                  replace=replace, ctx=ctx, weighted=True)
            return _npi.take(a, indices)
    else:
        if p is None:
            return _npi.choice(a=a, size=size, replace=replace, ctx=ctx, weighted=False, out=out)
        else:
            return _npi.choice(p, a=a, size=size, replace=replace, ctx=ctx, weighted=True, out=out)


def laplace(loc=0.0, scale=1.0, size=None, dtype=None, ctx=None, out=None):
    r"""Draw random samples from a Laplace distribution.

    Samples are distributed according to a Laplace distribution parametrized
    by *loc* (mean) and *scale* (the exponential decay).

    Parameters
    ----------
    loc : float, The position of the distribution peak.

    scale : float, the exponential decay.

    size : int or tuple of ints, optional. Output shape.
        If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Default is None, in which case a single value is returned.

    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output. Default is current context.
    out : ``ndarray``, optional
        Store output to an existing ``ndarray``.

    Returns
    -------
    out : _Symbol (symbol representing `mxnet.numpy.ndarray` in computational graphs)
        Drawn samples from the parameterized Laplace distribution.
    """
    from ._symbol import _Symbol as np_symbol
    input_type = (isinstance(loc, np_symbol), isinstance(scale, np_symbol))
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    if input_type == (True, True):
        return _npi.laplace(loc, scale, loc=None, scale=None, size=size,
                            ctx=ctx, dtype=dtype, out=out)
    elif input_type == (False, True):
        return _npi.laplace(scale, loc=loc, scale=None, size=size,
                            ctx=ctx, dtype=dtype, out=out)
    elif input_type == (True, False):
        return _npi.laplace(loc, loc=None, scale=scale, size=size,
                            ctx=ctx, dtype=dtype, out=out)
    else:
        return _npi.laplace(loc=loc, scale=scale, size=size,
                            ctx=ctx, dtype=dtype, out=out)


def gamma(shape, scale=1.0, size=None, dtype=None, ctx=None, out=None):
    """Draw samples from a Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    `shape` (sometimes designated "k") and `scale` (sometimes designated
    "theta"), where both parameters are > 0.

    Parameters
    ----------
    shape : float or array_like of floats
        The shape of the gamma distribution. Should be greater than zero.
    scale : float or array_like of floats, optional
        The scale of the gamma distribution. Should be greater than zero.
        Default is equal to 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``shape`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(shape, scale).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples.
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol
        Drawn samples from the parameterized gamma distribution.

    The Gamma distribution is often used to model the times to failure of
    electronic components, and arises naturally in processes for which the
    waiting times between Poisson distributed events are relevant.
    """
    from ._symbol import _Symbol as np_symbol
    input_type = (isinstance(shape, np_symbol), isinstance(scale, np_symbol))
    if ctx is None:
        ctx = current_context()
    if out is not None:
        size = out.shape
    if size == ():
        size = None
    if input_type == (True, True):
        return _npi.gamma(shape, scale, shape=None, scale=None, size=size,
                          ctx=ctx, dtype=dtype, out=out)
    elif input_type == (False, True):
        return _npi.gamma(scale, shape=shape, scale=None, size=size,
                          ctx=ctx, dtype=dtype, out=out)
    elif input_type == (True, False):
        return _npi.gamma(shape, shape=None, scale=scale, size=size,
                          ctx=ctx, dtype=dtype, out=out)
    else:
        return _npi.gamma(shape=shape, scale=scale, size=size,
                          ctx=ctx, dtype=dtype, out=out)

    raise ValueError("Distribution parameters must be either _Symbol or numbers")


def rayleigh(scale=0.0, size=None, ctx=None, out=None):
    r"""Draw samples from a Rayleigh distribution.
    The :math:`\chi` and Weibull distributions are generalizations of the
    Rayleigh.
    Parameters
    ----------
    scale : float or _Symbol
        Scale, also equals the mode. Must be non-negative. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``scale`` is a scalar.  Otherwise,
        ``np.array(scale).size`` samples are drawn.
    ctx : Context, optional
        Device context of output. Default is current context.
    Returns
    -------
    out : _Symbol
        Drawn samples from the parameterized Rayleigh distribution.
    """
    from ..numpy import _Symbol as np_symbol
    tensor_type_name = np_symbol
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    is_tensor = isinstance(scale, tensor_type_name)
    if is_tensor:
        return _npi.rayleigh(scale, scale=None, size=size, ctx=ctx, out=out)
    else:
        return _npi.rayleigh(scale=scale, size=size, ctx=ctx, out=out)


def beta(a, b, size=None, dtype=None, ctx=None):
    r"""Draw samples from a Beta distribution.

    The Beta distribution is a special case of the Dirichlet distribution,
    and is related to the Gamma distribution.  It has the probability
    distribution function

    .. math:: f(x; a,b) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}
                                                     (1 - x)^{\beta - 1},

    where the normalisation, B, is the beta function,

    .. math:: B(\alpha, \beta) = \int_0^1 t^{\alpha - 1}
                                 (1 - t)^{\beta - 1} dt.

    It is often seen in Bayesian inference and order statistics.

    Parameters
    ----------
    a : float or _Symbol of floats
        Alpha, positive (>0).
    b : float or _Symbol of floats
        Beta, positive (>0).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` and ``b`` are both scalars.
        Otherwise, ``np.broadcast(a, b).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples.
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
        Dtype 'float32' or 'float64' is strongly recommended,
        since lower precision might lead to out of range issue.
    ctx : Context, optional
        Device context of output. Default is current context.

    Notes
    -------
    To use this  operator with scalars as input, please run ``npx.set_np()`` first.

    Returns
    -------
    out : _Symbol
        Drawn samples from the parameterized beta distribution.
    """
    if dtype is None:
        dtype = np.float64 if is_np_default_dtype() else np.float32
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    # use fp64 to prevent precision loss
    X = gamma(a, 1, size=size, dtype='float64', ctx=ctx)
    Y = gamma(b, 1, size=size, dtype='float64', ctx=ctx)
    out = X/(X + Y)
    return out.astype(dtype)


def f(dfnum, dfden, size=None, ctx=None):
    r"""Draw samples from an F distribution.

    Samples are drawn from an F distribution with specified parameters,
    `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
    freedom in denominator), where both parameters must be greater than
    zero.

    The random variate of the F distribution (also known as the
    Fisher distribution) is a continuous probability distribution
    that arises in ANOVA tests, and is the ratio of two chi-square
    variates.

    Parameters
    ----------
    dfnum : float or _Symbol of floats
        Degrees of freedom in numerator, must be > 0.
    dfden : float or _Symbol of float
        Degrees of freedom in denominator, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``dfnum`` and ``dfden`` are both scalars.
        Otherwise, ``np.broadcast(dfnum, dfden).size`` samples are drawn.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol
        Drawn samples from the parameterized Fisher distribution.
    """
    X = chisquare(df=dfnum, size=size, ctx=ctx)
    Y = chisquare(df=dfden, size=size, ctx=ctx)
    return (X * dfden) / (Y * dfnum)


def chisquare(df, size=None, dtype=None, ctx=None):
    r"""
    chisquare(df, size=None, dtype=None, ctx=None)

    Draw samples from a chi-square distribution.

    When `df` independent random variables, each with standard normal
    distributions (mean 0, variance 1), are squared and summed, the
    resulting distribution is chi-square (see Notes).  This distribution
    is often used in hypothesis testing.

    Parameters
    ----------
    df : float or _Symbol of floats
         Number of degrees of freedom, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``df`` is a scalar.  Otherwise,
        ``np.array(df).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples.
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol
        Drawn samples from the parameterized chi-square distribution.

    Raises
    ------
    ValueError
        When `df` <= 0 or when an inappropriate `size`
        is given.

    Notes
    -----
    The variable obtained by summing the squares of `df` independent,
    standard normally distributed random variables:

    .. math:: Q = \sum_{i=0}^{\mathtt{df}} X^2_i

    is chi-square distributed, denoted

    .. math:: Q \sim \chi^2_k.

    The probability density function of the chi-squared distribution is

    .. math:: p(x) = \frac{(1/2)^{k/2}}{\Gamma(k/2)}
                     x^{k/2 - 1} e^{-x/2},

    where :math:`\Gamma` is the gamma function,

    .. math:: \Gamma(x) = \int_0^{-\infty} t^{x - 1} e^{-t} dt.

    References
    ----------
    .. [1] NIST "Engineering Statistics Handbook"
           https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    """
    if dtype is None:
        dtype = np.float64 if is_np_default_dtype() else np.float32
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    return gamma(df/2, 2, size=size, dtype=dtype, ctx=ctx)


def exponential(scale=1.0, size=None, ctx=None, out=None):
    r"""Draw samples from an exponential distribution.

    Parameters
    ----------
    scale : float or array_like of floats
        The scale parameter, :math:`\beta = 1/\lambda`. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``scale`` is a scalar.  Otherwise,
        ``np.array(scale).size`` samples are drawn.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : _Symbol (symbol representing `mxnet.numpy.ndarray` in computational graphs)
        Drawn samples from the parameterized exponential distribution.
    """
    from ..numpy import _Symbol as np_symbol
    tensor_type_name = np_symbol
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    is_tensor = isinstance(scale, tensor_type_name)
    if is_tensor:
        return _npi.exponential(scale, scale=None, size=size,
                                ctx=ctx, out=out)
    else:
        return _npi.exponential(scale=scale, size=size, ctx=ctx, out=out)


def weibull(a, size=None, ctx=None, out=None):
    r"""Draw samples from a 1-parameter Weibull distribution with given parameter a
    via inversion.

    Parameters
    ----------
    a : float or array_like of floats
        Shape of the distribution. Must be non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar. Otherwise,
        ``np.array(a).size`` samples are drawn.

    Returns
    -------
    out : _Symbol
        Drawn samples from the 1-parameter Weibull distribution.

    Examples
    --------
    >>> np.random.weibull(a=5)
    array(0.9553641)

    >>> np.random.weibull(a=5, size=[2,3])
    array([[1.0466299 , 1.1320982 , 0.98415005],
          [1.1430776 , 0.9532727 , 1.1344457 ]])

    >>> np.random.weibull(a=np.array([2,3])
    array([0.98843634, 1.0125613 ])

    The Weibull distribution is one of a class of Generalized Extreme
    Value (GEV) distributions. This class includes the Gumbel and Frechet
    distributions.

    The probability density for the Weibull distribution is
    f(x) = \frac{a}{\lambda}(\frac{x}{\lambda})^{a-1}e^{-(x/\lambda)^a},
    where a is the shape and \lambda the scale. The generated 1-parameter Weibull
    sample has the scale parameter \lambda = 1.

    The Weibull distribution is commonly used in reliability engineering to
    model time to failure, in modeling particle sizes, in information retrieval
    to model dwell time on pages, in quantitative finance to model risk etc.
    """
    from ..numpy import _Symbol as np_symbol
    tensor_type_name = np_symbol
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    is_tensor = isinstance(a, tensor_type_name)
    if is_tensor:
        return _npi.weibull(a, a=None, size=size, ctx=ctx, out=out)
    else:
        return _npi.weibull(a=a, size=size, ctx=ctx, out=out)


def pareto(a, size=None, ctx=None, out=None):
    r"""Draw samples from a Pareto II or Lomax distribution with specified shape a.

    Parameters
    ----------
    a : float or array_like of floats
            Shape of the distribution. Must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar. Otherwise,
        ``np.array(a).size`` samples are drawn.

    Returns
    -------
    out : _Symbol
        Drawn samples from the Pareto distribution.

    Examples
    --------
    >>> np.random.pareto(a=5)
    array(0.12749612)
    >>> mx.numpy.random.pareto(a=5, size=[2,3])
    array([[0.06933999, 0.0344373 , 0.10654891],
            [0.0311172 , 0.12911797, 0.03370714]])
    >>> np.random.pareto(a=np.array([2,3])
    array([0.26636696, 0.15685666])

    The probability density for the Pareto distribution is f(x) = \frac{am^a}{x^{a+1}}
    where a is the shape and m the scale. Here m is assumed 1. The Pareto distribution
    is a power law distribution. Pareto created it to describe the wealth in the economy.
    """
    from ..numpy import _Symbol as np_symbol
    tensor_type_name = np_symbol
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    is_tensor = isinstance(a, tensor_type_name)
    if is_tensor:
        return _npi.pareto(a, a=None, size=size, ctx=ctx, out=out)
    else:
        return _npi.pareto(a=a, size=size, ctx=ctx, out=out)


def power(a, size=None, ctx=None, out=None):
    r"""Draw samples in [0, 1] from a power distribution with given parameter a.

    Parameters
    ----------
    a : float or array_like of floats
        Shape of the distribution. Must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar. Otherwise,
        ``np.array(a).size`` samples are drawn.

    Returns
    -------
    out : _Symbol
        Drawn samples from the power distribution.

    Examples
    --------
    >>> np.random.power(a=5)
    array(0.8602478)
    >>> np.random.power(a=5, size=[2,3])
    array([[0.988391  , 0.5153122 , 0.9383134 ],
           [0.9078098 , 0.87819266, 0.730635]])
    >>> np.random.power(a=np.array([2,3])
    array([0.7499419 , 0.88894516])

    The probability density function is f(x; a) = ax^{a-1}, 0 \le x \le 1, a>0.
    The power distribution is just the inverse of the Pareto distribution and
    a special case of the Beta distribution.
    """
    from ..numpy import _Symbol as np_symbol
    tensor_type_name = np_symbol
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    is_tensor = isinstance(a, tensor_type_name)
    if is_tensor:
        return _npi.powerd(a, a=None, size=size, ctx=ctx, out=out)
    else:
        return _npi.powerd(a=a, size=size, ctx=ctx, out=out)


def multivariate_normal(mean, cov, size=None, check_valid=None, tol=None):
    """
    multivariate_normal(mean, cov, size=None, check_valid=None, tol=None)

    Draw random samples from a multivariate normal distribution.

    The multivariate normal, multinormal or Gaussian distribution is a
    generalization of the one-dimensional normal distribution to higher
    dimensions.  Such a distribution is specified by its mean and
    covariance matrix.  These parameters are analogous to the mean
    (average or "center") and variance (standard deviation, or "width,"
    squared) of the one-dimensional normal distribution.

    This operator is a little different from the one in official NumPy.
    The official NumPy operator only accepts 1-D ndarray as mean and 2-D ndarray as cov,
    whereas the operator in MXNet np supports batch operation and auto-broadcasting.

    Both `mean` and `cov` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.

    Parameters
    ----------
    mean : K-D _Symbol, of shape (..., N)
        Mean of the N-dimensional distribution.
    cov : (K+1)-D _Symbol, of shape (..., N, N)
        Covariance matrix of the distribution. The last two dimensions must be symmetric and
        positive-semidefinite for proper sampling.
    size : int or tuple of ints, optional
        Given a shape of, for example, ``(m,n,k)``,
        ``m*n*k`` identically distributed batchs of samples are
        generated, and packed in an `m`-by-`n`-by-`k` arrangement.
        If no shape is specified, a batch of (`N`-D) sample is returned.
    check_valid : { 'warn', 'raise', 'ignore' }, optional
        Behavior when the covariance matrix is not positive semidefinite.
        (Not supported)
    tol : float, optional
        Tolerance when checking the singular values in covariance matrix.
        cov is cast to double before the check.
        (Not supported)

    Returns
    -------
    out : _Symbol
        The input shape of `mean` and `cov` should satisfy the requirements of broadcasting.
        If the parameter `size` is not provided,
        the output shape is ``np.broadcast(mean.shape, cov.shape[:-1])``.
        Otherwise, the output shape is ``size + np.broadcast(mean.shape, cov.shape[:-1])``

    Examples
    --------
    >>> mean = np.array([1, 2])
    >>> cov = np.array([[1, 0], [0, 1]])
    >>> x = np.random.multivariate_normal(mean, cov, (3, 3))
    >>> x.shape
    (3, 3, 2)

    The following is probably true, given that 0.6 is roughly twice the
    standard deviation:

    >>> list((x[0,0,:] - mean) < 0.6)
    [True, True] # random

    # Performs autobroadcasting when the batch shape of
    # `mean` and `cov` is different but compatible.

    >>> mean = np.zeros((3,2)) # shape (3, 2)
    >>> cov = np.array([[1, 0], [0, 100]]) # shape (2, 2)
    >>> x = np.random.multivariate_normal(mean, cov)
    >>> x
    array([[-1.6115597 , -8.726251  ],
           [ 2.2425299 ,  2.8104177 ],
           [ 0.36229908, -8.386591  ]])
    """
    if check_valid is not None:
        raise NotImplementedError('Parameter `check_valid` is not supported')
    if tol is not None:
        raise NotImplementedError('Parameter `tol` is not supported')
    return _npi.mvn_fallback(mean, cov, size=size)


def shuffle(x):
    """
    Modify a sequence in-place by shuffling its contents.

    This function only shuffles the array along the first axis of a
    multi-dimensional array. The order of sub-arrays is changed but
    their contents remain the same.

    Parameters
    ----------
    x: _Symbol
        The array or list to be shuffled.

    Returns
    -------
    None

    Examples
    --------
    >>> arr = np.arange(10)
    >>> np.random.shuffle(arr)
    >>> arr
    array([5., 1., 0., 6., 7., 3., 9., 8., 4., 2.])  # random

    Multi-dimensional arrays are only shuffled along the first axis:

    >>> arr = np.arange(9).reshape((3, 3))
    >>> np.random.shuffle(arr)
    >>> arr
    array([[6., 7., 8.], # random
           [3., 4., 5.],
           [0., 1., 2.]])
    """
    _npi.shuffle(x, out=x)


def geometric(p, size=None, dtype=None, ctx=None, out=None):
    """Draw samples from the geometric distribution.

    Bernoulli trials are experiments with one of two outcomes: success or failure
    (an example of such an experiment is flipping a coin). The geometric distribution
    models the number of trials that must be run in order to achieve success.
    It is therefore supported on the positive integers, k = 1, 2, ....

    Parameters
    ----------
    p : float or array_like of floats. The probability of success of an individual trial.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn. If size is None (default), a single value is returned if p is a scalar.
        Otherwise, np.array(p).size samples are drawn.
    ctx : Context, optional
        Device context of output. Default is current context.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized geometric distribution.

    The geometric distribution models the number of trials that must be run in order to achieve success.
    """
    from ._symbol import _Symbol as np_symbol
    input_type = isinstance(p, np_symbol)
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if size == ():
        size = None
    if input_type:
        return _npi.geometric(p, prob=None, size=size,
                              ctx=ctx, dtype=dtype, out=out)
    else:
        return _npi.geometric(prob=p, size=size,
                              ctx=ctx, dtype=dtype, out=out)
