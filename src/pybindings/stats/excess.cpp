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

#include <ndhist/stats/excess.hpp>

namespace bp = boost::python;

namespace ndhist {
namespace stats {

void register_excess()
{
    bp::def("excess"
      , &py::excess
      , ( bp::arg("hist")
        , bp::arg("axis")=bp::object()
        )
      , "Calculates the excess kurtosis along the given axis of the given    \n"
        "ndhist object. As in statistics, the excess kurtosis is defined as  \n"
        ":math:`Excess[x] = Kurtosis[x] - 3`.                                \n"
        "It is a measure how normaly distributed the distribution is.        \n"
        "This function generates a projection along the given axis and then  \n"
        "calculates the excess kurtosis.                                     \n"
        "If ``None`` is given as axis argument (the default), the excess     \n"
        "kurtosis for all individual axes of the ndhist object is calculated \n"
        "and returned as a tuple. But if the dimensionality of the histogram \n"
        "is 1, a scalar value is returned.                                   \n"
        "                                                                    \n"
        ".. note:: This function is only defined for ndhist objects with POD \n"
        "          type axis values AND POD type weight values.              \n"
    );
}

}// namespace stats
}// namespace ndhist
