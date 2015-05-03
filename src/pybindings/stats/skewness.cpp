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

#include <ndhist/stats/skewness.hpp>

namespace bp = boost::python;

namespace ndhist {
namespace stats {

void register_skewness()
{
    bp::def("skewness"
      , &py::skewness
      , ( bp::arg("hist")
        , bp::arg("axis")=bp::object()
        )
      , "Calculates the skewness along the given axis of the given                \n"
        "ndhist object. As in statistics, the skewness is defined as              \n"
        ":math:`SKEWNESS[x] = ( E[x^3] - 3 V[x] E[x] - E[x]^3 ) / \\sqrt{V[x]^3}`.\n"
        "This function generates a projection along the given axis and then       \n"
        "calculates the skewness.                                                 \n"
        "If ``None`` is given as axis argument (the default), the skewness        \n"
        "for all individual axes of the ndhist object is calculated and           \n"
        "returned as a tuple. But if the dimensionality of the histogram is       \n"
        "1, a scalar value is returned.                                           \n"
        "                                                                         \n"
        ".. note:: This function is only defined for ndhist objects with POD      \n"
        "          type axis values AND POD type weight values.                   \n"
    );
}

}// namespace stats
}// namespace ndhist
