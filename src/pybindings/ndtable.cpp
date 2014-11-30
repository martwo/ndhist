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

#include <ndhist/ndtable.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

void register_ndtable()
{
    bp::class_<ndtable, boost::shared_ptr<ndtable> >("ndtable"
        , "The ndtable class holds data of a multi-dimensional tabularized \n"
          "function."
        , bp::init<
              bn::ndarray const &
            , bn::dtype const &
          >((bp::arg("shape"), bp::arg("dtype")))
    )
    .add_property("data", bp::make_function(&ndtable::GetNDArray, bn::ndarray_accessor_return()
        , (bp::arg("self")))
        , "The ndarray holding the table data.")
    ;
}

}// namespace ndhist
