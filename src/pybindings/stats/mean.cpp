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
      , "Calculates the mean value along the given axis. It generates a      \n"
        "projection along the given axis and then calculates                 \n"
        "the mean axis value. In statistics the mean is also known as the    \n"
        "expectation value ``E[x]``.                                         \n"
        "If ``None`` is given as axis argument (the default), the mean value \n"
        "for all axes of the ndhist object is calculated and returned as a   \n"
        "tuple. But if the dimensionality of the histogram is 1, a scalar    \n"
        "value is returned.                                                  \n"
        "\n"
        ".. note:: This function is only defined for ndhist objects with POD \n"
        "          axis values AND POD weight values.                        \n"
    );
}

}// namespace stats
}// namespace ndhist
