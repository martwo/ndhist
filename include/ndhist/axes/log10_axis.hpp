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
#ifndef NDHIST_AXES_LOG10_AXIS_HPP_INCLUDED
#define NDHIST_AXES_LOG10_AXIS_HPP_INCLUDED 1

#include <ndhist/axis.hpp>
#include <ndhist/detail/value_transforms/log10.hpp>
#include <ndhist/axes/constant_bin_width_axis.hpp>
#include <ndhist/axes/generic_axis.hpp>

namespace ndhist {
namespace axes {

template <typename AxisValueType>
class Log10Axis
  : public ConstantBinWidthAxis<AxisValueType, ::ndhist::detail::value_transforms::log10<AxisValueType> >
{
  public:
    typedef AxisValueType
            axis_value_type;

    typedef ::ndhist::detail::value_transforms::log10<axis_value_type>
            value_transform_type;

    typedef ConstantBinWidthAxis<axis_value_type, value_transform_type>
            base;

    typedef Log10Axis<axis_value_type>
            type;

  public:
    Log10Axis(
        bn::ndarray const & edges
      , std::string const & label
      , std::string const & name
      , bool has_underflow_bin
      , bool has_overflow_bin
      , bool is_extendable
      , intptr_t extension_max_fcap
      , intptr_t extension_max_bcap
    )
      : base(
            edges
          , label
          , name
          , has_underflow_bin
          , has_overflow_bin
          , is_extendable
          , extension_max_fcap
          , extension_max_bcap
        )
    {
        // Set up the axis's function pointers that are specific for this class
        // type, i.e. the functions that create explicitly an object of this
        // class type.
        Axis::create_fct_   = &type::create;
        Axis::deepcopy_fct_ = &type::deepcopy;
    }

    /**
     * Copy constructor.
     */
    Log10Axis(Log10Axis const & other)
      : base(other)
    {}

    NDHIST_AXIS_STATIC_METHOD_CREATE()

    NDHIST_AXIS_STATIC_METHOD_DEEPCOPY()
};

// Meta function to select the correct axis class.
// In cases where the axis value type is boost::python::object, the
// Log10Axis template cannot be used due to its performed arithmetic
// operations with the ConstantBinWidthAxis template.
// In those cases, we need to fall back to the GenericAxis template,
// which requires only the comparsion operator to be implemented for the
// boost::python::object object.
template <typename AxisValueType>
struct Log10Axis_selector
{
    typedef Log10Axis<AxisValueType>
            type;
};

template <typename AxisValueType>
struct select_Log10Axis_type
{
    typedef typename boost::mpl::eval_if<
                boost::is_same<AxisValueType, boost::python::object>
              , GenericAxis_selector<AxisValueType>
              , Log10Axis_selector<AxisValueType>
              >::type
            type;
};

// In order to expose the Log10Axis to Python, we need a wrapper
// around the Log10Axis template to get rid of the template
// parameter to avoid several Python classes, one for each axis value type.
namespace py {

typedef PyExtendableAxisWrapper<select_Log10Axis_type>
        log10_axis;

}//namespace py

}//namespace axes
}//namespace ndhist

#endif // ! NDHIST_AXES_LOG10_AXIS_HPP_INCLUDED
