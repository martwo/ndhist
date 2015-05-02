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
#include <cmath>
#include <sstream>
#include <vector>

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/python.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/iterators/flat_iterator.hpp>
#include <boost/numpy/python/make_tuple_from_container.hpp>

#include <ndhist/detail/bin_iter_value_type_traits.hpp>
#include <ndhist/type_support.hpp>
#include <ndhist/detail/utils.hpp>
#include <ndhist/stats/mean.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

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

namespace detail {

bp::object
calc_axis_moment(
    ndhist const & h
  , intptr_t const n
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
           << "are not supported by the moment function!";
        throw TypeError(ss.str());
    }

    #define NDHIST_MULTPLEX(r, seq)                                             \
        if(   bn::dtype::equivalent(h.get_axes()[axis_idx]->get_dtype(), bn::dtype::get_builtin<BOOST_PP_SEQ_ELEM(0,seq)>())\
           && bn::dtype::equivalent(h.get_weight_dtype(), bn::dtype::get_builtin<BOOST_PP_SEQ_ELEM(1,seq)>())\
          )                                                                     \
        {                                                                       \
            BOOST_PP_SEQ_ELEM(0,seq) moment = ::ndhist::stats::detail::calc_axis_moment_impl<BOOST_PP_SEQ_ELEM(0,seq), BOOST_PP_SEQ_ELEM(1,seq)>(h, n, axis_idx);\
            return bp::object(moment);                                          \
        }
    BOOST_PP_SEQ_FOR_EACH_PRODUCT(NDHIST_MULTPLEX, (NDHIST_TYPE_SUPPORT_AXIS_VALUE_TYPES_WITHOUT_OBJECT)(NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES_WITHOUT_OBJECT))
    #undef NDHIST_MULTPLEX

    std::stringstream ss;
    ss << "The combination of axis value type and weight value type of this "
       << "ndhist object is not supported for the moment function!";
    throw TypeError(ss.str());
}

}// namespace detail

bp::object
moment(
    ndhist const & h
  , intptr_t const n
  , bp::object const & axis
)
{
    if(axis != bp::object())
    {
        // A particular axis is given. So calculate the moment only for that
        // axis and return a scalar.
        intptr_t const axis_idx = bp::extract<intptr_t>(axis);
        return detail::calc_axis_moment(h, n, axis_idx);
    }

    // No axis was specified, so calculate the moment for all axes.
    intptr_t const nd = h.get_nd();

    // Return a scalar value if the dimensionality of the histogram is 1.
    if(nd == 1)
    {
        return detail::calc_axis_moment(h, n, 0);
    }

    // Return a tuple holding the moment values for each single axis.
    std::vector<bp::object> moments;
    moments.reserve(nd);
    for(intptr_t i=0; i<nd; ++i)
    {
        moments.push_back(detail::calc_axis_moment(h, n, i));
    }
    return boost::python::make_tuple_from_container(moments.begin(), moments.end());
}

}// namespace py
}// namespace stats
}// namespace ndhist
