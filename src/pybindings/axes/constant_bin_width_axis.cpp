/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <ndhist@martin-wolf.org>
 *
 * This file is distributed under the BSD 2-Clause Open Source License
 * (See LICENSE file).
 *
 */
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/axis.hpp>
#include <ndhist/axes/constant_bin_width_axis.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace axes {

void register_constant_bin_width_axis()
{
    bp::class_<py::constant_bin_width_axis, bp::bases<Axis>, boost::shared_ptr<py::constant_bin_width_axis> >("constant_bin_width_axis"
        , "The constant_bin_width_axis class provides an axis class for axes "
          "with constant bin widths."
        , bp::init<
            bn::ndarray const &
          , std::string const &
          , std::string const &
          , bool
          , bool
          , intptr_t
          , intptr_t
          >(
          ( bp::arg("self")
          , bp::arg("edges")
          , bp::arg("label")=std::string("")
          , bp::arg("name")=std::string("")
          , bp::arg("has_oor_bins")=true
          , bp::arg("is_extendable")=false
          , bp::arg("extension_max_fcap")=0
          , bp::arg("extension_max_bcap")=0
          )
          )
        )
        .def(axis_pyinterface<py::constant_bin_width_axis>())
    ;
}

}//namespace axes
}//namespace ndhist
