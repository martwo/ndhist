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

#include <ndhist/stats/std.hpp>

namespace bp = boost::python;

namespace ndhist {

namespace stats {

void register_std()
{
    bp::def("std"
      , &py::std
      , ( bp::arg("hist")
        , bp::arg("axis")=bp::object()
        )
      , "Calculates the standard deviation (std) along the given axis of the \n"
        "given ndhist object. As in statistics, the standard deviation is    \n"
        "defined as :math:`\\sqrt{V[x]}`, where :math:`V[x]` is the variance.\n"
        "This function generates a projection along the given axis and then  \n"
        "calculates the standard deviation.                                  \n"
        "If ``None`` is given as axis argument (the default), the standard   \n"
        "deviation for all individual axes of the ndhist object is calculated\n"
        "and returned as a tuple. But if the dimensionality of the histogram \n"
        "is 1, a scalar value is returned.                                   \n"
        "                                                                    \n"
        ".. note:: This function is only defined for ndhist objects with POD \n"
        "          type axis values AND POD type weight values.              \n"
    );
}

}// namespace stats
}// namespace ndhist
