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

#include <ndhist/ndhist.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

void register_ndhist()
{
    bp::class_<ndhist, boost::shared_ptr<ndhist> >("ndhist"
        , "The ndhist class provides a multi-dimensional histogram class."
        , bp::init<
            bn::ndarray const &
          , bp::list const &
          , bn::dtype const &
          >(
          ( bp::arg("nbins")
          , bp::arg("edges")
          , bp::arg("dtype")
          )
          )
        )

        // We use the bn::ndarray_accessor_return CallPolicy to keep the
        // ndhist object alive as long as the returned ndarray is alive.
        .add_property("bc", bp::make_function(&ndhist::GetBinContentArray, bn::ndarray_accessor_return()
            , (bp::arg("self")))
            , "The ndarray holding the bin contents.")

        .def("get_bin_edges", bp::make_function(&ndhist::GetEdgesArray, bn::ndarray_accessor_return()
            , (bp::arg("self"), bp::arg("axis")=0))
            , "Gets the ndarray holding the bin edges for the given axis. "
              "The default axis is 0.")
    ;
}

}// namespace ndhist
