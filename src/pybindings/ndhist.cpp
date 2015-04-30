/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <ndhist@martin-wolf.org>
 *
 * This file is distributed under the BSD 2-Clause Open Source License
 * (See LICENSE file).
 *
 */
#include <boost/preprocessor/seq/for_each.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <boost/python/operators.hpp>

#include <boost/numpy/ndarray_accessor_return.hpp>
#include <boost/numpy/ndarray_accessor_tuple_return.hpp>
#include <boost/numpy/dstream.hpp>
#include <boost/numpy/utilities.hpp>

#include <ndhist/type_support.hpp>
#include <ndhist/ndhist.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

void register_ndhist()
{
    bp::class_<ndhist, boost::shared_ptr<ndhist> >("ndhist"
        , "The ndhist class provides a multi-dimensional histogram class."
        , bp::init<
            bp::tuple const &
          , bp::object const &
          , bp::object const &
          >(
          ( bp::arg("axes")
          , bp::arg("dtype")=bn::dtype::get_builtin<double>()
          , bp::arg("bc_class")=bp::object()
          )
          )
        )

        .add_property("ndim", &ndhist::get_nd
            , "The dimensionality of the histogram.")
        .add_property("shape", &ndhist::py_get_shape
            , "The tuple holding the shape of the histogram. A shape entry "
              "corresponds to the number of bins of each axis, "
              "including possible under- and overflow bins.")
        .add_property("axes", &ndhist::py_get_axes
            , "The tuple holding the axis objects of this histogram.")
        .add_property("nbins", &ndhist::py_get_nbins
            , "The tuple holding the number of bins (excluding the possible "
              "under- and overflow bins) for each axis.")
        .add_property("binedges", &ndhist::py_get_binedges
            , "The tuple holding ndarray objects with the bin edges for each "
              "axis. In case the histogram is 1-dimensional, just a single "
              "ndarray is returned.")
        .add_property("bincenters", &ndhist::py_get_bincenters
            , "The tuple holding ndarray objects with the bin centers for each "
              "axis. In case the histogram is 1-dimensional, just a single "
              "ndarray is returned.")
        .add_property("binwidths", &ndhist::py_get_binwidths
            , "The tuple holding ndarray objects with the bin widths for each "
              "axis. In case the histogram is 1-dimensional, just a single "
              "ndarray is returned.")

        .add_property("title", &ndhist::py_get_title, &ndhist::py_set_title
            , "The title of the histogram.")
        .add_property("labels", &ndhist::py_get_labels
            , "The tuple holding the labels of the axes.")
        .add_property("is_view", &ndhist::is_view
            , "The flag if this ndhist object is a view into the bin content "
              "array of an other ndhist object.")

        .add_property("base",
              &ndhist::py_get_base
            , "In case this ndhist object provides a data view into an other "
              "ndhist object, the base is a reference to this ndhist object.")

        // We use the bn::ndarray_accessor_return CallPolicy to keep the
        // ndhist object alive as long as the returned ndarray is alive.
        .add_property("binentries", bp::make_function(
              &ndhist::py_get_noe_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the bin entries (counts) for each bin. "
              "It excludes possible under- and overflow bins.")
        .add_property("full_binentries", bp::make_function(
              &ndhist::py_get_full_noe_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the bin entries (counts) for each bin. "
              "In contrast to the ``binentries`` property, it includes "
              "possible under- and overflow bins.")
        .add_property("bincontent", bp::make_function(
              &ndhist::py_get_sow_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the bin contents (sum of weights) for "
              "each bin. "
              "It excludes possible under- and overflow bins.")
        .add_property("full_bincontent", bp::make_function(
              &ndhist::py_get_full_sow_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the bin contents (sum of weights) for "
              "each bin. "
              "In contrast to the ``bincontent`` property, it includes "
              "possible under- and overflow bins.")
        .add_property("squaredweights", bp::make_function(
              &ndhist::py_get_sows_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the sum of the weights squared for each bin. "
              "It excludes possible under- and overflow bins.")
        .add_property("full_squaredweights", bp::make_function(
              &ndhist::py_get_full_sows_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the sum of the weights squared for each bin. "
              "In contrast to the ``squaredweights`` property, it includes "
              "possible under- and overflow bins.")
        .add_property("binerror"
            , &ndhist::py_get_binerror_ndarray
            , "The ndarray holding the square root of the bin's sum of weights "
              "squared, i.e. the bin error values.")

        //----------------------------------------------------------------------
        // Underflow and overflow properties.
        .add_property("underflow_entries"
            , &ndhist::py_get_underflow_entries
            , "The underflow (number of entries) bins for each dimension analog   \n"
              "to the ``underflow`` property.                                     \n"
              "See the documentation of the ``underflow`` property for more details.")
        .add_property("underflow_entries_view"
            , bp::make_function(
                  &ndhist::py_get_underflow_entries_view
                , bn::ndarray_accessor_tuple_return()
              )
            , "Same as underflow_entries but the returned ndarrays are actual     \n"
              "views into the histogram's bin content array.")
        .add_property("overflow_entries"
            , &ndhist::py_get_overflow_entries
            , "The overflow (number of entries) bins for each dimension analog    \n"
              "to the ``underflow`` property.                                     \n"
              "See the documentation of the ``underflow`` property for more details.")
        .add_property("overflow_entries_view"
            , bp::make_function(
                  &ndhist::py_get_overflow_entries_view
                , bn::ndarray_accessor_tuple_return()
              )
            , "Same as overflow_entries but the returned ndarrays are actual      \n"
              "views into the histogram's bin content array.")
        .add_property("underflow"
            , &ndhist::py_get_underflow
            , "A tuple of length *ndim* where each element is a *ndim*-dimensional\n"
              "ndarray holding the underflow (sum of weights) bins for the        \n"
              "particular axis, where the index of the tuple element specifies    \n"
              "the axis. The dimension of the particular axis is collapsed to     \n"
              "one and the lengths of the other dimensions are extended by two.   \n"
              "Each returned ndarray holds copies of the histogram bins.          \n"
              "                                                                   \n"
              "Example: For (3,2) shaped two-dimensional histogram, there will    \n"
              "         be two tuple elements with a two-dimensional ndarray      \n"
              "         each. The shape of the first array (i.e. for the first    \n"
              "         axis) will be (1,4) and the shape of the second array     \n"
              "         will be (5,1).")
        .add_property("underflow_view"
            , bp::make_function(
                  &ndhist::py_get_underflow_view
                , bn::ndarray_accessor_tuple_return()
              )
            , "See the documentation of the ``underflow`` property.               \n"
              "But each returned ndarray is an actual view into the internal bin  \n"
              "content array of the histogram.")
        .add_property("overflow"
            , &ndhist::py_get_overflow
            , "The overflow (sum-of-weights) bins for each dimension analog to the\n"
              "``underflow`` property.                                            \n"
              "See the documentation of the ``underflow`` property for more details.")
        .add_property("overflow_view"
            , bp::make_function(
                  &ndhist::py_get_overflow_view
                , bn::ndarray_accessor_tuple_return()
              )
            , "See the documentation of the ``overflow`` property.                \n"
              "But each returned ndarray is an actual view into the internal bin  \n"
              "content array of the histogram.")
        .add_property("underflow_squaredweights"
            , &ndhist::py_get_underflow_squaredweights
            , "The underflow (sum-of-weights-squared) bins for each dimension     \n"
              "analog to the ``underflow`` property.                              \n"
              "See the documentation of the ``underflow`` property for more details.")
        .add_property("underflow_squaredweights_view"
            , bp::make_function(
                  &ndhist::py_get_underflow_squaredweights_view
                , bn::ndarray_accessor_tuple_return()
              )
            , "Same as the underflow_squaredweights property, but the returned    \n"
              "ndarrays are actual views into the bin content array of the        \n"
              "histogram.")
        .add_property("overflow_squaredweights"
            , &ndhist::py_get_overflow_squaredweights
            , "The overflow (sum-of-weights-squared) bins for each dimension      \n"
              "analog to the ``underflow_squaredweights`` property.               \n"
              "See the documentation of the ``underflow_squaredweights`` property \n"
              "for more details.")
        .add_property("overflow_squaredweights_view"
            , bp::make_function(
                  &ndhist::py_get_overflow_squaredweights_view
                , bn::ndarray_accessor_tuple_return()
              )
            , "Same as the overflow_squaredweights property, but the returned     \n"
              "ndarrays are actual views into the bin content array of the        \n"
              "histogram.")

        .add_property("ndvalues_dtype", &ndhist::get_ndvalues_dtype
            , "The dtype object describing the ndvalues array needed for "
              "filling the histogram. This property can be used in the "
              "ndarray.view method in order to get a view on a MxN array to "
              "fill it into a N-dimensional histogram with M entries.")

        .add_property("weight_dtype"
            , &ndhist::get_weight_dtype
            , "The dtype object describing the data type of the weight values.")
        .add_property("has_object_weight_dtype"
            , &ndhist::weight_type_is_object
            , "The flag if the weight data type is a Python object and not a "
              "POD type.")

        .add_property("MAX_TUPLE_FILL_NDIM", &ndhist::get_max_tuple_fill_nd
            , "The maximal dimensionality of the histogram, which "
              "is still supported for filling with a tuple of arrays as "
              "ndvalue function argument. Otherwise a structured ndarray needs "
              "to be used as ndvalue argument.")

        .def("clear", &ndhist::clear
            , (bp::arg("self"))
            , "Sets all bins of this ndhist object to zero. If this ndhist "
              "object is a view, only the bins of the view are set to zero. "
              "This allows to set only a selection of bins to zero.")

        .def("deepcopy", &ndhist::deepcopy
            , (bp::arg("self"))
            , "Copies this ndhist object. It copies also the underlaying data, "
              "even if this ndhist object is a view.")

        .def("get_binedges", &ndhist::get_binedges_ndarray
            , (bp::arg("self"), bp::arg("axis")=0)
            , "Gets the ndarray holding the bin edges of the given axis. "
              "The default axis is 0.")
        .def("get_bincenters", &ndhist::get_bincenters_ndarray
            , (bp::arg("self"), bp::arg("axis")=0)
            , "Gets the ndarray holding the bin centers of the given axis. "
              "The default axis is 0.")
        .def("fill", &ndhist::py_fill
            , (bp::arg("ndvalues"), bp::arg("weight")=bp::object())
            , "Fills the histogram with the given n-dimensional numbers, "
              "weighted by the given weights. If no weights are specified, "
              "``1`` will be used for each entry.")
        .def("empty_like", &ndhist::empty_like
            , (bp::arg("self"))
            , "Creates a new empty ndhist object having the same binning and "
              "data types as this ndhist object.")
        .def("is_compatible", &ndhist::is_compatible
            , (bp::arg("self"), bp::arg("other"))
            , "Checks if the given ndhist object is compatible with this "
              "ndhist object. This means, the dimensionality and the edges of "
              "all axes must match.")
        .def("project", &ndhist::project
            , (bp::arg("self"), bp::arg("dims"))
            , "Create a new ndhist object that is the projection of this        \n"
              "histogram containing only the dimensions, which have been        \n"
              "specified through the *dims* argument.                           \n"
              "All other dimensions are collapsed (summed) accordingly into     \n"
              "the remaining specified dimensions.                              \n")

        .def("merge_axis_bins", &ndhist::merge_axis_bins
            , ( bp::arg("self")
              , bp::arg("axis")
              , bp::arg("nbins_to_merge")=2
              , bp::arg("copy")=true
              )
            , "Merges the specified number of bins of the specified axis.\n"
              "\n"
              "If the number of bins to merge is not a true divisor of the "
              "number of bins (without possible under- and overflow bins), the "
              "remaining bins will be put into the overflow bin. If the axis "
              "did not have an overflow bin, a new overflow bin will be "
              "created. But if the axis is extendable, the remaining bins "
              "will be discarded, because an extendable axis cannot hold an "
              "overflow bin by definition.\n"
              "\n"
              "If this ndhist object is a data view into an other ndhist "
              "object, or the optional argument *copy* is set to ``True``, "
              "a deep copy of this ndhist object is created first. Otherwise "
              "the rebin operation is performed directly on this ndhist "
              "object itself.\n"
              "\n"
              "The default for nbins_to_merge is 2.\n"
              "The default for copy is ``True``.\n"
              "\n"
              "It returns the changed (this or the copy) ndhist object.")

        .def("merge_bins"
          , (boost::shared_ptr<ndhist> (ndhist::*)(bp::tuple const &, bp::tuple const &, bool const))&ndhist::merge_bins
          , ( bp::arg("self")
            , bp::arg("axes")
            , bp::arg("nbins_to_merge")
            , bp::arg("copy")=true
            )
          , "Same as the ``merge_axis_bins`` method but allows to merge bins "
            "of several axes at once.")

        // Slicing.
        .def("__getitem__", &ndhist::operator[]
            , (bp::arg("self"), bp::arg("arg"))
            , "Slices the histogram based on the given slicing argument. "
              "It follows the numpy slicing rules for basic indexing.")

        // Arithmetic operator overloads.
        .def(bp::self += bp::self)
        .def(bp::self + bp::self)
        #define NDHIST_WEIGHT_VALUE_TYPE_SUPPORT(r, data, WEIGHT_VALUE_TYPE)    \
            .def(bp::self *= WEIGHT_VALUE_TYPE ())                              \
            .def(bp::self /= WEIGHT_VALUE_TYPE ())                              \
            .def(bp::self * WEIGHT_VALUE_TYPE ())                               \
            .def(WEIGHT_VALUE_TYPE () * bp::self)                               \
            .def(bp::self / WEIGHT_VALUE_TYPE ())
        BOOST_PP_SEQ_FOR_EACH(NDHIST_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES_WITHOUT_OBJECT)
        #undef NDHIST_WEIGHT_VALUE_TYPE_SUPPORT
    ;
    //bp::implicitly_convertible< boost::shared_ptr<ndhist>, boost::shared_ptr<ndhist const> >();
}

}// namespace ndhist
