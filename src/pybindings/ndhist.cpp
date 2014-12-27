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

#include <boost/numpy/ndarray_accessor_return.hpp>
#include <boost/numpy/dstream.hpp>

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
          , bn::dtype const &
          , bp::object
          >(
          ( bp::arg("axes")
          , bp::arg("dtype")
          , bp::arg("bc_class")=bp::object()
          )
          )
        )

        .add_property("nd", &ndhist::get_nd
            , "The dimensionality of the histogram.")
        .add_property("nbins", &ndhist::py_get_nbins
            , "The tuple holding the number of bins for each axis.")

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
        .add_property("squaredweights", bp::make_function(
              &ndhist::py_get_sows_ndarray
            , bn::ndarray_accessor_return())
            , "The ndarray holding the sum of weights for for each bin.")

        .add_property("ndvalues_dtype", &ndhist::get_ndvalues_dtype
            , "The dtype object describing the ndvalues array needed for "
              "filling the histogram. This property can be used in the "
              "ndarray.view method in order to get a view on a MxN array to "
              "fill it into a N-dimensional histgram with M entries.")

        .add_property("max_tuple_fill_nd", &ndhist::get_max_tuple_fill_nd
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
    ;
}

}// namespace ndhist
