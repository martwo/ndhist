from __future__ import division

import math
import numpy as np

from ndhist.core import generic_axis, linear_axis, log10_axis

def linear(start, stop
  , width=1
  , label=''
  , name=''
  , add_underflow_bin=True
  , add_overflow_bin=True
  , extend=False
  , extracap=0
):
    """Creates a linear axis with bins in the range [``start``, ``stop``]
    having a constant bin width of ``width``.

    The number of bins is ceil'ed in cases where ``width`` is not an integral
    divisor of (``stop`` - ``start``).

    :type  start: float
    :param start: The start of the axis range [start, stop].

    :type  stop: float
    :param stop: The stop of the axis range [start, stop].

    :type  width: float
    :param width: The constant width of the bins.

    :type  label: str
    :param label: The label of the axis.

    :type  name: str
    :param name: The name of the axis. It is used to name the axis in a
        structured numpy ndarray when filling ndvalues through a structured
        array.

    :type  add_underflow_bin: bool
    :param add_underflow_bin: The switch, if an underflow bin should be added
        to the edges array automatically.

    :type  add_overflow_bin: bool
    :param add_overflow_bin: The switch, if an overflow bin should be added
        to the edges array automatically.

    :type  extend: bool
    :param extend: The switch if the axis is extendable (True) or not (False).
        In case it is extendable, no under- and overflow bins will be added
        automatically and the range of the axis will extend automatically
        whenever values are filled that lie outside the current axis range.

    :type  extracap: int
    :param extracap: The number of extra bin capacity for the axis, in case the
        axis is extendable. A value greater than zero will allocate extra memory
        to reduce the number of required memory reallocations when the axis
        needs to get extended.

    """
    nbins = int(math.ceil((stop - start) / width)) + 1
    edges = np.linspace(start, stop, num=nbins, endpoint=True)

    # Add under- and overflow bin edges if the axis is not extendable.
    if(extend):
        add_underflow_bin = False
        add_overflow_bin = False
    else:
        if(add_underflow_bin):
            edges_new = np.empty((edges.size+1,), edges.dtype)
            edges_new[1:] = edges
            edges_new[0]  = -np.inf
            edges = edges_new
        if(add_overflow_bin):
            edges_new = np.empty((edges.size+1,), edges.dtype)
            edges_new[:-1] = edges
            edges_new[-1] = +np.inf
            edges = edges_new

    #print(edges)
    axis = linear_axis(edges, label, name, add_underflow_bin, add_overflow_bin, extend, extracap, extracap)
    return axis

def linear_bins(start, nbins
  , width=1
  , label=''
  , name=''
  , add_underflow_bin=True
  , add_overflow_bin=True
  , extend=False
  , extracap=0
):
    """Creates a linear axis with ``nbins`` starting from ``start`` and having
    the constant bin width of ``width``.

    :type  start: float
    :param start: The start of the axis range [start, stop].

    :type  nbins: int
    :param nbins: The number of bins (excluding the possible under- and overflow
        bins) for the axis.

    :type  width: float
    :param width: The constant width of the bins.

    :type  label: str
    :param label: The label of the axis.

    :type  name: str
    :param name: The name of the axis. It is used to name the axis in a
        structured numpy ndarray when filling ndvalues through a structured
        array.

    :type  add_underflow_bin: bool
    :param add_underflow_bin: The switch, if an underflow bin should be added
        to the edges array automatically.

    :type  add_overflow_bin: bool
    :param add_overflow_bin: The switch, if an overflow bin should be added
        to the edges array automatically.

    :type  extend: bool
    :param extend: The switch if the axis is extendable (True) or not (False).
        In case it is extendable, no under- and overflow bins will be added
        automatically and the range of the axis will extend automatically
        whenever values are filled that lie outside the current axis range.

    :type  extracap: int
    :param extracap: The number of extra bin capacity for the axis, in case the
        axis is extendable. A value greater than zero will allocate extra memory
        to reduce the number of required memory reallocations when the axis
        needs to get extended.

    """
    stop = start + nbins*width
    return linear(start, stop, width, label, name, add_underflow_bin, add_overflow_bin, extend, extracap)

def log10(start, stop
  , width=0.1
  , label=''
  , name=''
  , add_underflow_bin=True
  , add_overflow_bin=True
  , extend=False
  , extracap=0
):
    """Creates a logarithmic base 10 axis with bins in the range
    [``start``, ``stop``] having a constant log10 space bin width of ``width``.

    The number of bins is ceil'ed in cases where ``width`` is not an integral
    divisor of (``log10(stop)`` - ``log10(start)``).

    :type  start: float
    :param start: The start of the axis range [start, stop], e.g. ``0.1``.

    :type  stop: float
    :param stop: The stop of the axis range [start, stop], e.g. ``100``.

    :type  width: float
    :param width: The constant width of the bins in log10 space, e.g. ``0.1``
        that is ten bins per decade.

    :type  label: str
    :param label: The label of the axis.

    :type  name: str
    :param name: The name of the axis. It is used to name the axis in a
        structured numpy ndarray when filling ndvalues through a structured
        array.

    :type  add_underflow_bin: bool
    :param add_underflow_bin: The switch, if an underflow bin should be added
        to the edges array automatically.

    :type  add_overflow_bin: bool
    :param add_overflow_bin: The switch, if an overflow bin should be added
        to the edges array automatically.

    :type  extend: bool
    :param extend: The switch if the axis is extendable (True) or not (False).
        In case it is extendable, no under- and overflow bins will be added
        automatically and the range of the axis will extend automatically
        whenever values are filled that lie outside the current axis range.

    :type  extracap: int
    :param extracap: The number of extra bin capacity for the axis, in case the
        axis is extendable. A value greater than zero will allocate extra memory
        to reduce the number of required memory reallocations when the axis
        needs to get extended.

    """
    nbins = int(math.ceil((np.log10(stop) - np.log10(start)) / width)) + 1
    edges = np.logspace(np.log10(start), np.log10(stop), num=nbins, endpoint=True)

    # Add under- and overflow bin edges if the axis is not extendable.
    if(extend):
        add_underflow_bin = False
        add_overflow_bin = False
    else:
        if(add_underflow_bin):
            edges_new = np.empty((edges.size+1,), edges.dtype)
            edges_new[1:] = edges
            edges_new[0]  = 0
            edges = edges_new
        if(add_overflow_bin):
            edges_new = np.empty((edges.size+1,), edges.dtype)
            edges_new[:-1] = edges
            edges_new[-1] = +np.inf
            edges = edges_new

    #print(edges)
    axis = log10_axis(edges, label, name, add_underflow_bin, add_overflow_bin, extend, extracap, extracap)
    return axis
