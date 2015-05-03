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
#ifndef NDHIST_STATS_SKEWNESS_HPP_INCLUDED
#define NDHIST_STATS_SKEWNESS_HPP_INCLUDED 1

#include <cmath>

#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>
#include <ndhist/stats/moment.hpp>
#include <ndhist/stats/var.hpp>

namespace ndhist {
namespace stats {

namespace detail {

template <typename AxisValueType, typename WeightValueType>
double
calc_axis_skewness_impl(
    ndhist const & h
  , intptr_t const axis
)
{
    // Do the projection here, so it won't be done twice.
    ndhist const proj = (h.get_nd() == 1 ? h : h.project(bp::object(axis)));

    double moment1 = calc_axis_moment_impl<AxisValueType, WeightValueType>(proj, 1, axis);
    double moment3 = calc_axis_moment_impl<AxisValueType, WeightValueType>(proj, 3, axis);
    double var = calc_axis_var_impl<AxisValueType, WeightValueType>(proj, axis);

    // SKEWNESS[x] = ( E[x^3] - 3 V[x] E[x] - E[x]^3 ) / \sqrt{V[x]^3}
    return (moment3 - 3*var*moment1 - moment1*moment1*moment1) / std::sqrt(var*var*var);
}

}// namespace detail

namespace py {

/**
 * @brief Calculates the skewness along the given axis of the given
 *     ndhist object. As in statistics, the skewness is defined as
 *     :math:`SKEWNESS[x] = ( E[x^3] - 3 V[x] E[x] - E[x]^3 ) / \\sqrt{V[x]^3}`.
 *     This function generates a projection along the given axis and then
 *     calculates the skewness.
 *     If None is given as axis, the skewness for all individual axes of the
 *     ndhist object will be calculated and returned as a tuple. But if the
 *     dimensionality of the ndhist object is 1, a scalar value is returned.
 *
 * @note This function is only defined for ndhist objects with POD axis values
 *     AND POD weight values.
 */
boost::python::object
skewness(
    ndhist const & h
  , boost::python::object const & axis = boost::python::object()
);

}// namespace py

}// namespace stats
}// namespace ndhist

#endif // !NDHIST_STATS_SKEWNESS_HPP_INCLUDED
