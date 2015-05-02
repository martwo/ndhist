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

#include <ndhist/stats/moment.hpp>

namespace bp = boost::python;

namespace ndhist {

namespace stats {

void register_moment()
{
    bp::def("moment"
      , &py::moment
      , ( bp::arg("hist")
        , bp::arg("n")=1
        , bp::arg("axis")=bp::object()
        )
      , "Calculates the n'th moment value along the given axis of the given  \n"
        "ndhist object weighted by the sum of weights in each bin.           \n"
        "It generates a projection along the given axis and then calculates  \n"
        "the n'th moment axis value.                                         \n"
        "In statistics the n'th moment is defined as the expectation of x^n, \n"
        "i.e. ``E[x^n]``, where x is the bin center axis value in this case. \n"
        "If ``None`` is given as axis argument (the default), the n'th moment\n"
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
