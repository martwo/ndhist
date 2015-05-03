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

#include <cmath>

#include <boost/python.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/iterators/multi_flat_iterator.hpp>

#include <ndhist/detail/bin_iter_value_type_traits.hpp>
#include <ndhist/ndhist.hpp>

namespace ndhist {
namespace stats {

namespace detail {

/**
 * Calculates the n'th moment value along the given axis of the given ndhist
 * object weighted by the sum of weights in each bin.
 * It generates a projection along the given axis and then calculates the n'th
 * moment value.
 * In statistics the n'th moment is defined as the expectation of x^n, i.e.
 * ``E[x^n]``, where x is the bin center axis value.
 */
template <typename AxisValueType, typename WeightValueType>
AxisValueType
calc_axis_moment_impl(
    ndhist const & h
  , intptr_t const n
  , intptr_t const axis
)
{
    // Project the given histogram to the given axis (if nd > 1).
    ndhist const proj = (h.get_nd() == 1 ? h : h.project(bp::object(axis)));

    // Iterate over the bins (which are along the given axis) and exclude
    // possible under- and overflow bins.
    Axis const & theaxis = *proj.get_axes()[0];
    intptr_t nbins = theaxis.get_n_bins();
    if(theaxis.has_underflow_bin()) --nbins;
    if(theaxis.has_overflow_bin()) --nbins;
    bn::ndarray proj_bincenters_arr = theaxis.get_bincenters_ndarray();
    bn::ndarray proj_bc_arr = proj.bc_.construct_ndarray(proj.bc_.get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
    typedef bn::iterators::multi_flat_iterator<2>::impl<
                bn::iterators::single_value<AxisValueType>
              , ::ndhist::detail::bin_iter_value_type_traits<WeightValueType>
            >
            multi_iter_t;
    multi_iter_t iter(
        proj_bincenters_arr
      , proj_bc_arr
      , bn::detail::iter_operand::flags::READONLY::value
      , bn::detail::iter_operand::flags::READONLY::value
    );

    // Skip the underflow bin.
    if(theaxis.has_underflow_bin()) ++iter;

    AxisValueType moment = 0;
    WeightValueType sow_sum = 0;
    while(nbins > 0)
    {
        typename multi_iter_t::multi_references_type multi_value = *iter;
        typename multi_iter_t::value_ref_type_0 axis_bincenter_value = multi_value.value_0;
        typename multi_iter_t::value_ref_type_1 bin                  = multi_value.value_1;

        sow_sum += *bin.sow_;
        moment += *bin.sow_ * (n == 1 ? axis_bincenter_value
                            : (n == 2 ? axis_bincenter_value*axis_bincenter_value
                            : (n == 3 ? axis_bincenter_value*axis_bincenter_value*axis_bincenter_value
                            : std::pow(axis_bincenter_value, n))));

        ++iter;
        --nbins;
    }
    moment /= sow_sum;
    return moment;
}

}// namespace detail

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
