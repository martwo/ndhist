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
#include <boost/python.hpp>

#include <ndhist/stats/median.hpp>

namespace bp = boost::python;

namespace ndhist {

namespace stats {

void register_median()
{
    bp::def("median"
      , &py::median
      , ( bp::arg("hist")
        , bp::arg("axis")=bp::object()
        )
      , "Calculates the median value along the given axis of the given ndhist\n"
        "object.                                                             \n"
        "It generates a projection along the given axis and then calculates  \n"
        "the median value. The median value is defined as the axis value     \n"
        "where half of the sum of the weights are below and above that value.\n"
        "If ``None`` is given as axis argument (the default), the median     \n"
        "for all individual axes of the ndhist object is calculated and      \n"
        "returned as a tuple. But if the dimensionality of the histogram is  \n"
        "1, a scalar value is returned.                                      \n"
        "\n"
        ".. note:: This function is only defined for ndhist objects with POD \n"
        "          type axis values AND POD type weight values.              \n"
    );
}

}// namespace stats
}// namespace ndhist
