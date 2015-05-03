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
#ifndef NDHIST_STATS_EXCESS_HPP_INCLUDED
#define DHIST_STATS_EXCESS_HPP_INCLUDED 1

#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>
#include <ndhist/stats/kurtosis.hpp>

namespace ndhist {
namespace stats {

namespace detail {

template <typename AxisValueType, typename WeightValueType>
double
calc_axis_excess_impl(
    ndhist const & h
  , intptr_t const axis
)
{
    return calc_axis_kurtosis_impl<AxisValueType, WeightValueType>(h, axis) - 3;
}

}// namespace detail

namespace py {

/**
 * @brief Calculates the excess kurtosis along the given axis of the given
 *     ndhist object. As in statistics, the excess kurtosis is defined as
 *     :math:`Excess[x] = Kurtosis[x] - 3`.
 *     This function generates a projection along the given axis and then
 *     calculates the excess kurtosis.
 *     If None is given as axis, the excess kurtosis for all individual axes
 *     of the ndhist object will be calculated and returned as a tuple.
 *     But if the dimensionality of the ndhist object is 1, a scalar value is
 *     returned.
 *
 * @note This function is only defined for ndhist objects with POD axis values
 *     AND POD weight values.
 */
boost::python::object
excess(
    ndhist const & h
  , boost::python::object const & axis = boost::python::object()
);

}// namespace py

}// namespace stats
}// namespace ndhist

#endif // !DHIST_STATS_EXCESS_HPP_INCLUDED
