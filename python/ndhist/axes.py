from __future__ import division

import math
import numpy as np

from ndhist.core import constant_bin_width_axis

def linear(start, stop, width=1, label='', name='', addoorbins=True, extend=False, extracap=0):
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

    :type  addoorbins: bool
    :param addoorbins: The switch, if out-of-range (oor) bins should be added
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
        addoorbins = False
    else:
        if(addoorbins):
            edges_new = np.empty((edges.size+2,), edges.dtype)
            edges_new[1:-1] = edges
            edges_new[0]  = -np.inf
            edges_new[-1] = +np.inf
            edges = edges_new

    print(edges)
    axis = constant_bin_width_axis(edges, label, name, addoorbins, extend, extracap, extracap)
    return axis

def linear_bins(start, nbins, width=1, label='', name='', addoorbins=True, extend=False, extracap=0):
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

    :type  addoorbins: bool
    :param addoorbins: The switch, if out-of-range (oor) bins should be added
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
    return linear(start, stop, width, label, name, addoorbins, extend, extracap)
