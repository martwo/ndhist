import numpy as np

from ndhist import core

def _histsave_handler_hdf(h, f, where, name, overwrite):
    """Saves the given ndhist object to a hdf file using the tables package.

    """
    import tables

    # Open the HDF file for read/write if f is string holding the file name.
    close_file = False
    if(isinstance(f, str)):
        close_file = True
        f = tables.File(f, 'a')

    try:
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
    except:
        # On error, close the already opened file and re-raise the error.
        if(close_file):
            f.close()
        raise

    # Close the file, if we have opened it ourselves.
    if(close_file):
        f.close()

def _histload_handler_hdf(f, histgroup):
    """Loads a ndhist object from a hdf file using the tables package.

    """
    import tables

    # Open the HDF file for reading if f is string holding the file name.
    close_file = False
    if(isinstance(f, str)):
        close_file = True
        f = tables.File(f, 'r')

    try:
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
    except:
        # On error, close the already opened file and re-raise the error.
        if(close_file):
            f.close()
        raise

    # Close the file, if we have opened it ourselves.
    if(close_file):
        f.close()

    return h

def is_hdf_file(f):
    """Checks if the given file object is recognized as a HDF file.

    :type  f: str | tables.File
    :param f: The file object. Either a str object holding the file name or
        a HDF file instance.

    """
    import tables

    if((isinstance(f, str) and (f[-4:] == '.hdf' or f[-3:] == '.h5')) or
       (isinstance(f, tables.File))
      ):
        return True

    return False

def histsave(h, f, where, name, overwrite=False):
    """Saves the given ndhist object to the given file.

    It raises a NotImplementedError if the given file type is not supported.

    :type  h: ndhist
    :param h: The histogram object that should get stored.

    :type  f: str | tables.File
    :param f: The file instance into which the histogram should get saved.
        If this is a str instance holding the file name, the type of the file
        is determined from the file name extension string, e.g. ``".hdf"`` for
        HDF files.
        The following file extensions are supported for HDF files:

          - ``".hdf"``
          - ``".h5"``

    :type  where: str
    :param where: The parent group/location within the file where the histogram
        should get stored to.

        This argument can also be an instance of a group class, that is
        compatible with the given file type instance.

    :type  name: str
    :param name: The name of the data group for this histogram.

    :type  overwrite: bool
    :param overwrite: Flag if an existing histogram of the same name should get
        overwritten (``True``) or not (``False``).

    """
    if(not isinstance(h, core.ndhist)):
        raise TypeError(
            'The given histogram object is not an instance of the ndhist '+
            'class!')

    if(h.has_object_weight_dtype):
        raise TypeError(
            'Only ndhist objects with POD weight types can be stored to a file!')

    for (idx, axis) in enumerate(h.axes):
        if(axis.has_object_value_dtype):
            raise TypeError(
                'Only ndhist objects with POD axis value types can be stored '+
                'to a file. Axis "'+idx+'" does not have a POD value type!')

    try:
        if(is_hdf_file(f)):
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

    :type  f: str | tables.File
    :param f: The file instance from which the histogram should get loaded.
        If this is a str instance holding the file name, the type of the file
        is determined from the file name extension string, e.g. ``".hdf"`` for
        HDF files.
        The following file extensions are supported for HDF files:

          - ``".hdf"``
          - ``".h5"``

    :type  histgroup: str
    :param histgroup: The data group name within the file where the histogram is
        stored in. It's the group created by the histsave function.

        This argument can also be an instance of a group class, that is
        compatible with the given file type instance.

    """
    try:
        if(is_hdf_file(f)):
            return _histload_handler_hdf(f, histgroup)
    except ImportError:
        pass

    raise NotImplementedError(
        'The histogram could not be loaded from the given file. '+
        'No storage handler is available for the given file type!')
