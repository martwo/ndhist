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

#include <ndhist/stats/mean.hpp>

namespace bp = boost::python;

namespace ndhist {

namespace stats {

void register_mean()
{
    bp::def("mean"
      , &py::mean
      , ( bp::arg("hist")
        , bp::arg("axis")=bp::object()
        )
      , "Calculates the mean value along the given axis of the given ndhist  \n"
        "object.                                                             \n"
        "Since the mean is equal to the first order expectation, this        \n"
        "function just calls the ``ndhist.stats.expectation`` function to    \n"
        "calculate the first order expectation value.                        \n"
        "If ``None`` is given as axis argument (the default), the mean value \n"
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
