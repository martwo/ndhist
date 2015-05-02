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
#ifndef NDHIST_STATS_MOMENT_HPP_INCLUDED
#define NDHIST_STATS_MOMENT_HPP_INCLUDED 1

#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>

namespace ndhist {
namespace stats {
namespace py {

/**
 * @brief Calculates the n'th moment value along the given axis of the given
 *     ndhist object weighted by the sum of weights in each bin.
 *     It generates a projection along the given axis and then calculates the
 *     n'th moment value.
 *     In statistics the n'th moment is defined as the expectation of x^n, i.e.
 *     ``E[x^n]``, where x is the bin center axis value.
 *     If None is given as axis, the moment value for all axes of the ndhist
 *     object will be calculated and returned as a tuple. But if the
 *     dimensionality of the ndhist object is 1, a scalar value is returned.
 *
 * @note This function is only defined for ndhist objects with POD axis values
 *     AND POD weight values.
 */
boost::python::object
moment(
    ndhist const & h
  , intptr_t const n = 1
  , boost::python::object const & axis = boost::python::object()
);

}// namespace py
}// namespace stats
}// namespace ndhist

#endif // !NDHIST_STATS_MOMENT_HPP_INCLUDED
