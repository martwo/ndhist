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
#ifndef NDHIST_STATS_VAR_HPP_INCLUDED
#define NDHIST_STATS_VAR_HPP_INCLUDED 1

#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>
#include <ndhist/stats/expectation.hpp>

namespace ndhist {
namespace stats {

namespace detail {

template <typename AxisValueType, typename WeightValueType>
double
calc_axis_var_impl(
    ndhist const & h
  , intptr_t const axis
)
{
    // Do the projection here, so it won't be done twice.
    ndhist const proj = (h.get_nd() == 1 ? h : h.project(bp::object(axis)));

    double expectation1 = calc_axis_expectation_impl<AxisValueType, WeightValueType>(proj, 1, axis);
    double expectation2 = calc_axis_expectation_impl<AxisValueType, WeightValueType>(proj, 2, axis);

    // V[x] = E[x^2] - E[x]^2
    return expectation2 - expectation1*expectation1;
}

}// namespace detail

namespace py {

/**
 * @brief Calculates the variance along the given axis of the given
 *     ndhist object. As in statistics, the variance is defined as
 *     :math:`V[x] = E[x^2] - E[x]^2`.
 *     This function generates a projection along the given axis and then
 *     calculates the variance.
 *     If None is given as axis, the variance for all individual axes of the
 *     ndhist object will be calculated and returned as a tuple. But if the
 *     dimensionality of the ndhist object is 1, a scalar value is returned.
 *
 * @note This function is only defined for ndhist objects with POD axis values
 *     AND POD weight values.
 */
boost::python::object
var(
    ndhist const & h
  , boost::python::object const & axis = boost::python::object()
);

}// namespace py

}// namespace stats
}// namespace ndhist

#endif // !NDHIST_STATS_VAR_HPP_INCLUDED
