import numpy as np

from ndhist import core

def _histsave_handler_hdf(h, f, where, name, overwrite):
    """Saves the given ndhist object to a hdf file using the tables package.

    """
    import tables

    parent_group = f.get_node(where)

    if(name in parent_group._v_children):
        if(overwrite):
            f.remove_node(parent_group, name, recursive=True)
        else:
            raise KeyError(
                "A histogram with name '%s' already exists within "\
                "the group '%s'!"%(name, where))

    # Create a new group and store all necessary attributes and arrays into it.
    group = f.create_group(where, name)
    attrs = group._v_attrs

    # Save ndhist attributes.
    attrs['ndim'] = h.ndim
    attrs['title'] = h.title
    attrs['weight_dtype'] = h.weight_dtype.str

    def _save_array(arr, where):
        filters = tables.Filters(complib='blosc', complevel=9)
        ca = f.create_carray(group, where, tables.Atom.from_dtype(arr.dtype), arr.shape, filters=filters)
        ca[...] = arr

    # Save the axes information.
    for dim in range(0, h.ndim):
        axis = h.axes[dim]
        attrs['axis_%d_class_name'%(dim)] = axis.__class__.__name__
        _save_array(axis.binedges, 'axis_%d_binedges'%(dim))
        attrs['axis_%d_label'%(dim)]              = axis.label
        attrs['axis_%d_name'%(dim)]               = axis.name
        attrs['axis_%d_has_underflow_bin'%(dim)]  = axis.has_underflow_bin
        attrs['axis_%d_has_overflow_bin'%(dim)]   = axis.has_overflow_bin
        attrs['axis_%d_is_extendable'%(dim)]      = axis.is_extendable
        attrs['axis_%d_extension_max_fcap'%(dim)] = axis.extension_max_fcap
        attrs['axis_%d_extension_max_bcap'%(dim)] = axis.extension_max_bcap

    # Save the ndhist data arrays.
    _save_array(h.full_binentries, 'full_binentries')
    _save_array(h.full_bincontent, 'full_bincontent')
    _save_array(h.full_squaredweights, 'full_squaredweights')

def _histload_handler_hdf(f, histgroup):
    """Loads a ndhist object from a hdf file using the tables package.

    """
    import tables

    group = f.get_node(histgroup)
    attrs = group._v_attrs

    ndim = int(attrs['ndim'])
    title = str(attrs['title'])
    weight_dtype = np.dtype(attrs['weight_dtype'])

    def _load_array(name):
        return group._v_children[name].read()

    # Create the ndhist axis objects for all dimensions.
    axes = []
    for dim in range(0, ndim):
        axis_class_name = attrs['axis_%d_class_name'%(dim)]
        if(not axis_class_name in core.__dict__):
            raise TypeError(
                'The ndhist axis class "'+axis_class_name+'" is unkown! '+
                'Make sure that the versions of the stored data and the '
                'software match!')
        axis_class = core.__dict__[axis_class_name]

        binedges = _load_array('axis_%d_binedges'%(dim))
        label              = attrs['axis_%d_label'%(dim)]
        name               = attrs['axis_%d_name'%(dim)]
        has_underflow_bin  = attrs['axis_%d_has_underflow_bin'%(dim)]
        has_overflow_bin   = attrs['axis_%d_has_overflow_bin'%(dim)]
        is_extendable      = attrs['axis_%d_is_extendable'%(dim)]
        extension_max_fcap = attrs['axis_%d_extension_max_fcap'%(dim)]
        extension_max_bcap = attrs['axis_%d_extension_max_bcap'%(dim)]

        axes.append(axis_class(
            binedges
          , str(label)
          , str(name)
          , bool(has_underflow_bin)
          , bool(has_overflow_bin)
          , bool(is_extendable)
          , int(extension_max_fcap)
          , int(extension_max_bcap)
        ))

    # Create the (empty) ndhist object.
    h = core.ndhist(tuple(axes), dtype=weight_dtype)

    # Set the ndhist's title.
    h.title = title

    # Load and assign the data to the histogram.
    h.full_binentries[...]     = _load_array('full_binentries')
    h.full_bincontent[...]     = _load_array('full_bincontent')
    h.full_squaredweights[...] = _load_array('full_squaredweights')

    return h

def histsave(h, f, where, name, overwrite=False):
    """Saves the given ndhist object to the given file.

    It raises a NotImplementedError if the given file type is not supported.

    :type  h: ndhist
    :param h: The histogram object that should get stored.

    :type  f: tables.File
    :param f: The file instance into which the histogram should get saved.

    :type  where: str
    :param where: The parent group/location within the file where the histogram
        should get stored to.

        This argument can also be an instance of a group class, that is
        compatible with the given file type instance.

    :type  name: str
    :param name: The name of the group for this histogram.

    """
    if(not isinstance(h, core.ndhist)):
        raise TypeError(
            'The given histogram object is not an instance of the ndhist '+
            'class!')

    if(h.has_object_weight_dtype):
        raise TypeError(
            'Only ndhist objects with POD weight types can be stored to a file!')

    try:
        import tables
        if(isinstance(f, tables.File)):
            return _histsave_handler_hdf(h, f, where, name, overwrite)
    except ImportError:
        pass

    raise NotImplementedError(
        'The histogram could not be saved to the given file. No storage '+
        'handler is available for the given file type!')

def histload(f, histgroup):
    """Loads a ndhist object, that is stored within the given group within the
    given file.

    It raises a NotImplementedError if the given file type is not supported.

    :type  f: tables.File
    :param f: The file instance from which the histogram should get loaded.

    :type  histgroup: str
    :param histgroup: The data group name within the file where the histogram is
        stored in. It's the group created by the histsave function.

        This argument can also be an instance of a group class, that is
        compatible with the given file type instance.

    """
    try:
        import tables
        if(isinstance(f, tables.File)):
            return _histload_handler_hdf(f, histgroup)
    except ImportError:
        pass

    raise NotImplementedError(
        'The histogram could not be loaded from the given file. '+
        'No storage handler is available for the given file type!')
