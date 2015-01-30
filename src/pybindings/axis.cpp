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

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

void register_axis()
{
    bp::class_<Axis, boost::shared_ptr<Axis> >("axis"
        , "The axis class provides an axis base class for all axis classes "
          "used by by ndhist class."
        , bp::init<
            bn::dtype const &
          , std::string const &
          , std::string const &
          , bool
          , intptr_t
          , intptr_t
          >(
          ( bp::arg("dtype")
          , bp::arg("label")=std::string("")
          , bp::arg("name")=std::string("")
          , bp::arg("is_extendable")=false
          , bp::arg("extension_max_fcap")=0
          , bp::arg("extension_max_bcap")=0
          )
          )
        )
        .add_property("name", &Axis::get_name, &Axis::set_name
            , "The name of the axis. It is the name of the column in the "
              "structured ndarray, when filling values via a structured "
              "ndarray."
        )
        .add_property("is_extendable", &Axis::is_extendable
            , "Flag if the axis is extendable (True) or not (False)."
        )
    ;
}

}//namespace ndhist
