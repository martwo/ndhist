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
#ifndef NDHIST_STATS_KURTOSIS_HPP_INCLUDED
#define NDHIST_STATS_KURTOSIS_HPP_INCLUDED 1

#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>
#include <ndhist/stats/expectation.hpp>
#include <ndhist/stats/var.hpp>

namespace ndhist {
namespace stats {

namespace detail {

template <typename AxisValueType, typename WeightValueType>
double
calc_axis_kurtosis_impl(
    ndhist const & h
  , intptr_t const axis
)
{
    // Do the projection here, so it won't be done twice.
    ndhist const proj = (h.get_nd() == 1 ? h : h.project(bp::object(axis)));

    double const expact1 = calc_axis_expectation_impl<AxisValueType, WeightValueType>(proj, 1, axis);
    double const expact2 = calc_axis_expectation_impl<AxisValueType, WeightValueType>(proj, 2, axis);
    double const expact3 = calc_axis_expectation_impl<AxisValueType, WeightValueType>(proj, 3, axis);
    double const expact4 = calc_axis_expectation_impl<AxisValueType, WeightValueType>(proj, 4, axis);
    double const var = calc_axis_var_impl<AxisValueType, WeightValueType>(proj, axis);

    // Kurtosis[x] = (E[x^4] - 4 E[x] E[x^3] + 6 E[x]^2 E[x^2] - 3 E[x]^4) / V[x]^2
    return (expact4 - 4*expact1*expact3 + 6*expact1*expact1*expact2 - 3*expact1*expact1*expact1*expact1) / (var*var);
}

}// namespace detail

namespace py {

/**
 * @brief Calculates the kurtosis along the given axis of the given
 *     ndhist object. As in statistics, the kurtosis is defined as
 *     :math:`Kurtosis[x] = (E[x^4] - 4 E[x] E[x^3] + 6 E[x]^2 E[x^2] - 3 E[x]^4) / V[x]^2`.
 *     This function generates a projection along the given axis and then
 *     calculates the kurtosis.
 *     If None is given as axis, the kurtosis for all individual axes of the
 *     ndhist object will be calculated and returned as a tuple. But if the
 *     dimensionality of the ndhist object is 1, a scalar value is returned.
 *
 * @note This function is only defined for ndhist objects with POD axis values
 *     AND POD weight values.
 */
boost::python::object
kurtosis(
    ndhist const & h
  , boost::python::object const & axis = boost::python::object()
);

}// namespace py

}// namespace stats
}// namespace ndhist

#endif // NDHIST_STATS_KURTOSIS_HPP_INCLUDED
