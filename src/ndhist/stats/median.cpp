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
#include <sstream>
#include <vector>

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/python.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/iterators/flat_iterator.hpp>
#include <boost/numpy/iterators/multi_flat_iterator.hpp>
#include <boost/numpy/python/make_tuple_from_container.hpp>

#include <ndhist/detail/bin_iter_value_type_traits.hpp>
#include <ndhist/type_support.hpp>
#include <ndhist/detail/utils.hpp>
#include <ndhist/stats/median.hpp>

namespace ndhist {
namespace stats {

namespace detail {

template <typename AxisValueType, typename WeightValueType>
AxisValueType
calc_axis_median_impl(
    ndhist const & h
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
    bn::ndarray proj_bc_arr = proj.bc_.construct_ndarray(proj.bc_.get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);

    // First calculate the median sum of weights sum.
    typedef bn::iterators::flat_iterator<
                ::ndhist::detail::bin_iter_value_type_traits<WeightValueType>
            >
            bin_iter_t;
    bin_iter_t bin_iter(
        proj_bc_arr
      , bn::detail::iter_operand::flags::READONLY::value
    );
    // Skip the underflow bin.
    if(theaxis.has_underflow_bin()) ++bin_iter;
    bool all_bins_are_zero = true;
    double median_sow_sum = 0;
    for(intptr_t i=0; i<nbins; ++i)
    {
        typename bin_iter_t::value_ref_type bin = *bin_iter;
        median_sow_sum += *bin.sow_;
        if(median_sow_sum != 0)
        {
            all_bins_are_zero = false;
        }
        ++bin_iter;
    }
    median_sow_sum *= 0.5;
    if(all_bins_are_zero)
    {
        std::stringstream ss;
        ss << "All bin content values are zero. The median value is ambiguous "
           << "in that case!";
        throw ValueError(ss.str());
    }

    // At least one bin content value is unequal than zero, search for the
    // median axis value.
    typedef bn::iterators::multi_flat_iterator<3>::impl<
                bn::iterators::single_value<AxisValueType>
              , bn::iterators::single_value<AxisValueType>
              , ::ndhist::detail::bin_iter_value_type_traits<WeightValueType>
            >
            multi_iter_t;
    bn::ndarray axis_bincenters_arr = theaxis.get_bincenters_ndarray();
    bn::ndarray axis_upper_binedges_arr = theaxis.get_upper_binedges_ndarray();
    multi_iter_t iter(
        axis_bincenters_arr
      , axis_upper_binedges_arr
      , proj_bc_arr
      , bn::detail::iter_operand::flags::READONLY::value
      , bn::detail::iter_operand::flags::READONLY::value
      , bn::detail::iter_operand::flags::READONLY::value
    );
    // Skip the underflow bin.
    if(theaxis.has_underflow_bin()) ++iter;
    WeightValueType curr_sow_sum = 0;
    for(intptr_t i=0; i<nbins; ++i)
    {
        typename multi_iter_t::multi_references_type multi_value = *iter;
        typename multi_iter_t::value_ref_type_0 axis_bincenter_value     = multi_value.value_0;
        typename multi_iter_t::value_ref_type_1 axis_upper_binedge_value = multi_value.value_1;
        typename multi_iter_t::value_ref_type_2 bin                      = multi_value.value_2;

        curr_sow_sum += *bin.sow_;
        if(curr_sow_sum == median_sow_sum)
        {
            return axis_upper_binedge_value;
        }
        else if(curr_sow_sum > median_sow_sum)
        {
            return axis_bincenter_value;
        }

        ++iter;
    }

    std::stringstream ss;
    ss << "The median value for axis '"<<axis<<"' could not be determined. "
       << "This is a BUG!";
    throw RuntimeError(ss.str());
}

}// namespace detail

namespace py {

namespace detail {

bp::object
calc_axis_median(
    ndhist const & h
  , intptr_t const axis
)
{
    // Determine the correct axis index.
    intptr_t const axis_idx = ::ndhist::detail::adjust_axis_index(h.get_nd(), axis);

    // Check that the axis and weight types are not bp::object.
    if(   h.get_axes()[axis_idx]->has_object_value_dtype()
       || h.has_object_weight_dtype()
    )
    {
        std::stringstream ss;
        ss << "The axis and weight data types must be POD types. Non-POD types "
           << "are not supported by the median function!";
        throw TypeError(ss.str());
    }

    #define NDHIST_MULTPLEX(r, seq)                                             \
        if(   bn::dtype::equivalent(h.get_axes()[axis_idx]->get_dtype(), bn::dtype::get_builtin<BOOST_PP_SEQ_ELEM(0,seq)>())\
           && bn::dtype::equivalent(h.get_weight_dtype(), bn::dtype::get_builtin<BOOST_PP_SEQ_ELEM(1,seq)>())\
          )                                                                     \
        {                                                                       \
            BOOST_PP_SEQ_ELEM(0,seq) median = ::ndhist::stats::detail::calc_axis_median_impl<BOOST_PP_SEQ_ELEM(0,seq), BOOST_PP_SEQ_ELEM(1,seq)>(h, axis_idx);\
            return bp::object(median);                                          \
        }
    BOOST_PP_SEQ_FOR_EACH_PRODUCT(NDHIST_MULTPLEX, (NDHIST_TYPE_SUPPORT_AXIS_VALUE_TYPES_WITHOUT_OBJECT)(NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES_WITHOUT_OBJECT))
    #undef NDHIST_MULTPLEX

    std::stringstream ss;
    ss << "The combination of axis value type and weight value type of this "
       << "ndhist object is not supported for the median function!";
    throw TypeError(ss.str());
}

}// namespace detail

bp::object
median(
    ndhist const & h
  , bp::object const & axis
)
{
    if(axis != bp::object())
    {
        // A particular axis is given. So calculate the median only for that
        // axis and return a scalar.
        intptr_t const axis_idx = bp::extract<intptr_t>(axis);
        return detail::calc_axis_median(h, axis_idx);
    }

    // No axis was specified, so calculate the median for all axes.
    intptr_t const nd = h.get_nd();

    // Return a scalar value if the dimensionality of the histogram is 1.
    if(nd == 1)
    {
        return detail::calc_axis_median(h, 0);
    }

    // Return a tuple holding the moment values for each single axis.
    std::vector<bp::object> medians;
    medians.reserve(nd);
    for(intptr_t i=0; i<nd; ++i)
    {
        medians.push_back(detail::calc_axis_median(h, i));
    }
    return boost::python::make_tuple_from_container(medians.begin(), medians.end());
}

}// namespace py
}// namespace stats
}// namespace ndhist
