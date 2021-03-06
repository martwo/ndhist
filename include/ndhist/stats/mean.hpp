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
#ifndef NDHIST_STATS_MEAN_HPP_INCLUDED
#define NDHIST_STATS_MEAN_HPP_INCLUDED 1

#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>
#include <ndhist/stats/expectation.hpp>

namespace ndhist {
namespace stats {

namespace detail {

template <typename AxisValueType, typename WeightValueType>
AxisValueType
calc_axis_mean_impl(
    ndhist const & h
  , intptr_t const axis
)
{
    return calc_axis_expectation_impl<AxisValueType, WeightValueType>(h, 1, axis);
}

}// namespace detail

namespace py {

/**
 * @brief Calculates the mean value along the given axis of the given ndhist
 *     object.
 *     Since the mean is equal to the first order expectation, this function
 *     just calls the expectation function to calculate the first order
 *     expectation value.
 *     If None is given as axis, the mean value for all axes of the ndhist
 *     object will be calculated and returned as a tuple. But if the
 *     dimensionality of the ndhist object is 1, a scalar value is returned.
 *
 * @note This function is only defined for ndhist objects with POD type axis
 *     values AND POD type weight values.
 */
boost::python::object
mean(
    ndhist const & h
  , boost::python::object const & axis = boost::python::object()
);

}// namespace py
}// namespace stats
}// namespace ndhist

#endif // !NDHIST_STATS_MEAN_HPP_INCLUDED
