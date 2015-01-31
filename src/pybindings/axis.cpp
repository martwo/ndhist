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

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

void register_axis()
{
    bp::class_<Axis, boost::shared_ptr<Axis> >("axis"
        , "The axis class provides an axis base class for all axis classes "
          "used by ndhist class."
        , bp::init<
            bn::ndarray const &
          , std::string const &
          , std::string const &
          , bool
          , intptr_t
          , intptr_t
          >(
          ( bp::arg("self")
          , bp::arg("edges")
          , bp::arg("label")=std::string("")
          , bp::arg("name")=std::string("")
          , bp::arg("is_extendable")=false
          , bp::arg("extension_max_fcap")=0
          , bp::arg("extension_max_bcap")=0
          )
          )
        )
        .def(axis_pyinterface<Axis>())
    ;
}

}//namespace ndhist
