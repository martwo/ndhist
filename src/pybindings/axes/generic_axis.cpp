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
#include <ndhist/axes/generic_axis.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace axes {

void register_generic_axis()
{
    bp::class_<py::generic_axis, bp::bases<Axis>, boost::shared_ptr<py::generic_axis> >("generic_axis"
        , "The generic_axis class provides an axis for axes with non-constant "
          "bin widths. Due to the generic bin nature of this axis, it is not "
          "extendable at all. So the under- and overflow bin edges must always "
          "be provided."
        , bp::init<
            bn::ndarray const &
          , std::string const &
          , std::string const &
          >(
          ( bp::arg("self")
          , bp::arg("edges")
          , bp::arg("label")=std::string("")
          , bp::arg("name")=std::string("")
          )
          )
        )
        .def(axis_pyinterface<py::generic_axis>())
    ;
}

}//namespace axes
}//namespace ndhist
