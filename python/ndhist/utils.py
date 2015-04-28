import numpy as np

def ndzip(h, *args):
    """Creates a structured ndarray holding the values for each dimension of the
    given ndhist histogram for each entry to fill. This function is useful when
    the dimensionality of the histogram is bigger than the supported histogram
    dimensionality for filling using a tuple of arrays.

    In case the number of arguments is less or equal to the maximal number of
    dimensions supported for tuple filling, the argument tuple is just returned
    to trigger the tuple filling procedure.
    Otherwise a structured ndarray is created.

    """
    if(len(args) != h.ndim):
        raise ValueError(
            'The number of arguments must be equal to the dimensionality '+
            'of the histogram')

    # Trigger the tuple fill if possible, which is avoid the creation of an
    # intermediate ndvalues array.
    if(len(args) <= h.MAX_TUPLE_FILL_NDIM):
        return args

    # Determine the number of entries.
    nentries = None
    for arg in args:
        a = np.array(arg)
        if(nentries is None):
            nentries = a.size
        else:
            if(a.size == 1):
                # Scalars can be broadcasted.
                continue
            if(a.size != nentries):
                if(nentries == 1):
                    # The previous arguments of size 1 can be broadcasted to
                    # this new size.
                    nentries = a.size
                else:
                    raise ValueError(
                        'The number of elements in each argument must be '+
                        'equal or 1!')

    ndvalues = np.empty(nentries, dtype=h.ndvalues_dtype)
    for (i, axis) in enumerate(h.axes):
        ndvalues[axis.name] = args[i]

    return ndvalues
