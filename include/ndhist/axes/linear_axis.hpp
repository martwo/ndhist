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
#ifndef NDHIST_AXES_LINEAR_AXIS_HPP_INCLUDED
#define NDHIST_AXES_LINEAR_AXIS_HPP_INCLUDED 1

#include <ndhist/detail/value_transforms/identity.hpp>
#include <ndhist/axes/constant_bin_width_axis.hpp>
#include <ndhist/axes/generic_axis.hpp>

namespace ndhist {
namespace axes {

template <typename AxisValueType>
class LinearAxis
  : public ConstantBinWidthAxis<AxisValueType, ::ndhist::detail::value_transforms::identity<AxisValueType> >
{
  public:
    typedef AxisValueType
            axis_value_type;

    typedef ::ndhist::detail::value_transforms::identity<axis_value_type>
            value_transform_type;

    typedef ConstantBinWidthAxis<axis_value_type, value_transform_type>
            base;

    typedef LinearAxis<axis_value_type>
            type;

  public:
    LinearAxis(
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
        base::create_fct_   = &type::create;
        base::deepcopy_fct_ = &type::deepcopy;
    }

    /**
     * Copy constructor.
     */
    LinearAxis(LinearAxis const & other)
      : base(other)
    {}

    static
    boost::shared_ptr<Axis>
    create(
        boost::numpy::ndarray const & edges
      , std::string const & label
      , std::string const & name
      , bool has_underflow_bin
      , bool has_overflow_bin
      , bool is_extendable
      , intptr_t extension_max_fcap
      , intptr_t extension_max_bcap
    )
    {
        return boost::shared_ptr<Axis>(new type(
            edges
          , label
          , name
          , has_underflow_bin
          , has_overflow_bin
          , is_extendable
          , extension_max_fcap
          , extension_max_bcap
        ));
    }

    static
    boost::shared_ptr<Axis>
    deepcopy(Axis const & axisbase)
    {
        type const & axis = *static_cast<type const *>(&axisbase);
        return boost::shared_ptr<Axis>(new type(axis));
    }
};

// Meta function to select the correct axis class.
// In cases where the axis value type is boost::python::object, the
// LinearAxis template cannot be used due to its performed arithmetic
// operations with the ConstantBinWidthAxis template.
// In those cases, we need to fall back to the GenericAxis template,
// which requires only the comparsion operator to be implemented for the
// boost::python::object object.
template <typename AxisValueType>
struct LinearAxis_selector
{
    typedef LinearAxis<AxisValueType>
            type;
};

template <typename AxisValueType>
struct select_LinearAxis_type
{
    typedef typename boost::mpl::eval_if<
                boost::is_same<AxisValueType, boost::python::object>
              , GenericAxis_selector<AxisValueType>
              , LinearAxis_selector<AxisValueType>
              >::type
            type;
};

// In order to expose the LinearAxis to Python, we need a wrapper
// around the LinearAxis template to get rid of the template
// parameter to avoid several Python classes, one for each axis value type.
namespace py {

typedef PyExtendableAxisWrapper<select_LinearAxis_type>
        linear_axis;

}//namespace py

}//namespace axes
}//namespace ndhist

#endif // ! NDHIST_AXES_LINEAR_AXIS_HPP_INCLUDED
