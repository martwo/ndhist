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
#ifndef NDHIST_STATS_MEDIAN_HPP_INCLUDED
#define NDHIST_STATS_MEDIAN_HPP_INCLUDED 1

#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>

namespace ndhist {
namespace stats {
namespace py {

/**
 * @brief Calculates the median value along the given axis of the given ndhist
 *     object.
 *     It generates a projection along the given axis and then calculates the
 *     median value.
 *     The median value is defined as the axis value where half of the sum of
 *     the weights are below and above that value.
 *     If None is given as axis, the median value for all axes of the ndhist
 *     object will be calculated and returned as a tuple. But if the
 *     dimensionality of the ndhist object is 1, a scalar value is returned.
 *
 * @note This function is only defined for ndhist objects with POD type axis
 *     values AND POD type weight values.
 */
boost::python::object
median(
    ndhist const & h
  , boost::python::object const & axis = boost::python::object()
);

}// namespace py
}// namespace stats
}// namespace ndhist

#endif // !NDHIST_STATS_MEDIAN_HPP_INCLUDED
