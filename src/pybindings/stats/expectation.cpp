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

#include <ndhist/stats/expectation.hpp>

namespace bp = boost::python;

namespace ndhist {

namespace stats {

void register_expectation()
{
    bp::def("expectation"
      , &py::expectation
      , ( bp::arg("hist")
        , bp::arg("n")=1
        , bp::arg("axis")=bp::object()
        )
      , "Calculates the n'th order expectation value along the given axis of \n"
        "the given ndhist object weighted by the sum of weights in each bin. \n"
        "It generates a projection along the given axis and then calculates  \n"
        "the n'th order expectation axis value.                              \n"
        "In statistics the n'th order expectation is defined as the          \n"
        "expectation of x^n, i.e. ``E[x^n]``, where x is the bin center axis \n"
        "value in this case.                                                 \n"
        "If ``None`` is given as axis argument (the default), the n'th order \n"
        "expectation value for all individual axes of the ndhist object is   \n"
        "calculated and returned as a tuple. But if the dimensionality of    \n"
        "the histogram is 1, a scalar value is returned.                     \n"
        "                                                                    \n"
        ".. note:: This function is only defined for ndhist objects with POD \n"
        "          type axis values AND POD type weight values.              \n"
    );
}

}// namespace stats
}// namespace ndhist
