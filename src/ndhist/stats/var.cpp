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

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/python/make_tuple_from_container.hpp>

#include <ndhist/error.hpp>
#include <ndhist/ndhist.hpp>
#include <ndhist/type_support.hpp>
#include <ndhist/detail/utils.hpp>
#include <ndhist/stats/var.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace stats {
namespace py {

namespace detail {

bp::object
calc_axis_var(
    ndhist const & h
  , intptr_t const axis
)
{
    // Determine the correct axis index and get the Axis object.
    intptr_t const axis_idx = ::ndhist::detail::adjust_axis_index(h.get_nd(), axis);
    Axis const & theaxis = *h.get_axes()[axis_idx];

    // Check that the axis and weight types are not bp::object.
    if(   theaxis.has_object_value_dtype()
       || h.has_object_weight_dtype()
    )
    {
        std::stringstream ss;
        ss << "The axis and weight data types must be POD types. Non-POD types "
           << "are not supported by the var function!";
        throw TypeError(ss.str());
    }

    #define NDHIST_MULTPLEX(r, seq)                                             \
        if(   bn::dtype::equivalent(theaxis.get_dtype(), bn::dtype::get_builtin<BOOST_PP_SEQ_ELEM(0,seq)>())\
           && bn::dtype::equivalent(h.get_weight_dtype(), bn::dtype::get_builtin<BOOST_PP_SEQ_ELEM(1,seq)>())\
          )                                                                     \
        {                                                                       \
            BOOST_PP_SEQ_ELEM(0,seq) var = ::ndhist::stats::detail::calc_axis_var_impl<BOOST_PP_SEQ_ELEM(0,seq), BOOST_PP_SEQ_ELEM(1,seq)>(h, axis_idx);\
            return bp::object(var);                                             \
        }
    BOOST_PP_SEQ_FOR_EACH_PRODUCT(NDHIST_MULTPLEX, (NDHIST_TYPE_SUPPORT_AXIS_VALUE_TYPES_WITHOUT_OBJECT)(NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES_WITHOUT_OBJECT))
    #undef NDHIST_MULTPLEX

    std::stringstream ss;
    ss << "The combination of axis value type and weight value type of this "
       << "ndhist object is not supported for the var function!";
    throw TypeError(ss.str());
}

}// namespace detail

bp::object
var(
    ndhist const & h
  , bp::object const & axis
)
{
    if(axis != bp::object())
    {
        // A particular axis is given. So calculate the variance only for that
        // axis and return a scalar.
        intptr_t const axis_idx = bp::extract<intptr_t>(axis);
        return detail::calc_axis_var(h, axis_idx);
    }

    // No axis was specified, so calculate the variance for all axes.
    intptr_t const nd = h.get_nd();

    // Return a scalar value if the dimensionality of the histogram is 1.
    if(nd == 1)
    {
        return detail::calc_axis_var(h, 0);
    }

    // Return a tuple holding the variance values for each single axis.
    std::vector<bp::object> vars;
    vars.reserve(nd);
    for(intptr_t i=0; i<nd; ++i)
    {
        vars.push_back(detail::calc_axis_var(h, i));
    }
    return boost::python::make_tuple_from_container(vars.begin(), vars.end());
}

}// namespace py
}// namespace stats
}// namespace ndhist
