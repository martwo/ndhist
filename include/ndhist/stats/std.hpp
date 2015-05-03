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
#ifndef NDHIST_STATS_STD_HPP_INCLUDED
#define NDHIST_STATS_STD_HPP_INCLUDED 1

#include <cmath>

#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>
#include <ndhist/stats/var.hpp>

namespace ndhist {
namespace stats {

namespace detail {

template <typename AxisValueType, typename WeightValueType>
AxisValueType
calc_axis_std_impl(
    ndhist const & h
  , intptr_t const axis
)
{
    return std::sqrt(calc_axis_var_impl<AxisValueType, WeightValueType>(h, axis));
}

}// namespace detail

namespace py {

/**
 * @brief Calculates the standard deviation (std) along the given axis of the
 *     given ndhist object. As in statistics, the standard deviation is defined
 *     as :math:`\sqrt{V[x]}`, where :math:`V[x]` is the variance.
 *     This function generates a projection along the given axis and then
 *     calculates the standard deviation.
 *     If None is given as axis, the standard deviation for all individual axes
 *     of the ndhist object will be calculated and returned as a tuple.
 *     But if the dimensionality of the ndhist object is 1, a scalar value is
 *     returned.
 *
 * @note This function is only defined for ndhist objects with POD axis values
 *     AND POD weight values.
 */
boost::python::object
std(
    ndhist const & h
  , boost::python::object const & axis = boost::python::object()
);

}// namespace py

}// namespace stats
}// namespace ndhist

#endif // !NDHIST_STATS_STD_HPP_INCLUDED
