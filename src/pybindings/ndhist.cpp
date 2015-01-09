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
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <boost/python/operators.hpp>

#include <boost/numpy/ndarray_accessor_return.hpp>
#include <boost/numpy/dstream.hpp>
#include <boost/numpy/utilities.hpp>

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
          , bp::object
          >(
          ( bp::arg("axes")
          , bp::arg("dtype")=bn::dtype::get_builtin<double>()
          , bp::arg("bc_class")=bp::object()
          )
          )
        )

        .add_property("ndim", &ndhist::get_nd
            , "The dimensionality of the histogram.")
        .add_property("nbins", &ndhist::py_get_nbins
            , "The tuple holding the number of bins for each axis.")
        .add_property("binedges", &ndhist::py_get_binedges
            , "The tuple holding ndarray objects with the bin edges for each "
              "axis. In case the histogram is 1-dimensional, just a single "
              "ndarray is returned.")

        .add_property("title", &ndhist::py_get_title, &ndhist::py_set_title
            , "The title of the histgram.")
        .add_property("labels", &ndhist::py_get_labels
            , "The tuple holding the labels of the axes.")

        // We use the bn::ndarray_accessor_return CallPolicy to keep the
        // ndhist object alive as long as the returned ndarray is alive.
        .add_property("binentries", bp::make_function(
              &ndhist::py_get_noe_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the bin entries (counts) for each bin.")
        .add_property("bincontent", bp::make_function(
              &ndhist::py_get_sow_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the bin contents (sum of weights) for "
              "each bin.")
        .add_property("_h_bincontent", bp::make_function(
              &ndhist::py_get_oorpadded_sow_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the bin contents (sum of weights) for "
              "each bin. The number of bins for each axis is increased by 2 "
              "(one bin on each side of the axis) holding the under- and "
              "overflow bins.")
        .add_property("squaredweights", bp::make_function(
              &ndhist::py_get_sows_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the sum of weights for each bin.")
        .add_property("_h_squaredweights", bp::make_function(
              &ndhist::py_get_oorpadded_sows_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the sum of weights for each bin."
              "The number of bins for each axis is increased by 2 "
              "(one bin on each side of the axis) holding the under- and "
              "overflow bins.")

        //----------------------------------------------------------------------
        // Out-of-range properties.
        .add_property("underflow_entries", &ndhist::py_get_underflow_entries
            , "The underflow (number of entries) bins for each dimension analog   \n"
              "to the ``underflow`` property.                                     \n"
              "See the documentation of the ``underflow`` property for more details.")
        .add_property("overflow_entries", &ndhist::py_get_overflow_entries
            , "The overflow (number of entries) bins for each dimension analog    \n"
              "to the ``underflow`` property.                                     \n"
              "See the documentation of the ``underflow`` property for more details.")
        .add_property("underflow", &ndhist::py_get_underflow
            , "A tuple of length *ndim* where each element is a *ndim*-dimensional\n"
              "ndarray holding the underflow (sum of weights) bins for the        \n"
              "particular axis, where the index of the tuple element specifies    \n"
              "the axis. The dimension of the particular axis is collapsed to     \n"
              "one and the lengths of the other dimensions are extended by two.   \n"
              "                                                                   \n"
              "Example: For (3,2) shaped two-dimensional histogram, there will    \n"
              "         be two tuple elements with a two-dimensional ndarray      \n"
              "         each. The shape of the first array (i.e. for the first    \n"
              "         axis) will be (1,4) and the shape of the second array     \n"
              "         will be (5,1).")
        .add_property("overflow", &ndhist::py_get_overflow
            , "The overflow (sum-of-weights) bins for each dimension analog to the\n"
              "``underflow`` property.                                            \n"
              "See the documentation of the ``underflow`` property for more details.")
        .add_property("underflow_squaredweights", &ndhist::py_get_underflow_squaredweights
            , "The underflow (sum-of-weights-squared) bins for each dimension     \n"
              "analog to the ``underflow`` property.                              \n"
              "See the documentation of the ``underflow`` property for more details.")
        .add_property("overflow_squaredweights", &ndhist::py_get_overflow_squaredweights
            , "The overflow (sum-of-weights-squared) bins for each dimension      \n"
              "analog to the ``underflow`` property.                              \n"
              "See the documentation of the ``underflow`` property for more details.")


        .add_property("ndvalues_dtype", &ndhist::get_ndvalues_dtype
            , "The dtype object describing the ndvalues array needed for "
              "filling the histogram. This property can be used in the "
              "ndarray.view method in order to get a view on a MxN array to "
              "fill it into a N-dimensional histgram with M entries.")

        .add_property("max_tuple_fill_ndim", &ndhist::get_max_tuple_fill_nd
            , "The maximal dimensionality of the histogram, which "
              "is still supported for filling with a tuple of arrays as "
              "ndvalue function argument. Otherwise a structured array needs "
              "to be used as ndvalue argument.")

        .def("get_bin_edges", &ndhist::get_edges_ndarray
            , (bp::arg("self"), bp::arg("axis")=0)
            , "Gets the ndarray holding the bin edges for the given axis. "
              "The default axis is 0.")
        .def("fill", &ndhist::fill
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

        // Arithmetic operator overloads.
        .def(bp::self += bp::self)
        .def(bp::self + bp::self)
        .def(bp::self *= double())
        .def(bp::self * double())
        .def(bp::self /= double())
        .def(bp::self / double())
    ;
}

}// namespace ndhist
