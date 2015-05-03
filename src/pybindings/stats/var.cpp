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

#include <ndhist/stats/var.hpp>

namespace bp = boost::python;

namespace ndhist {

namespace stats {

void register_var()
{
    bp::def("var"
      , &py::var
      , ( bp::arg("hist")
        , bp::arg("axis")=bp::object()
        )
      , "Calculates the variance along the given axis of the given ndhist    \n"
        "object. As in statistics, the variance is defined as                \n"
        ":math:`V[x] = E[x^2] - E[x]^2`.                                     \n"
        "This function generates a projection along the given axis and then  \n"
        "calculates the variance.                                            \n"
        "If ``None`` is given as axis argument (the default), the variance   \n"
        "for all individual axes of the ndhist object is calculated and      \n"
        "returned as a tuple. But if the dimensionality of the histogram is  \n"
        "1, a scalar value is returned.                                      \n"
        "                                                                    \n"
        ".. note:: This function is only defined for ndhist objects with POD \n"
        "          type axis values AND POD type weight values.              \n"
    );
}

}// namespace stats
}// namespace ndhist
