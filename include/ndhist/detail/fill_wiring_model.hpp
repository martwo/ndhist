/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <ndhist@martin-wolf.org>
 *
 * This file is distributed under the BSD 2-Clause Open Source License
 * (See LICENSE file).
 *
 */
#ifndef NDHIST_DETAIL_FILL_WIRING_MODEL_HPP_INCLUDED
#define NDHIST_DETAIL_FILL_WIRING_MODEL_HPP_INCLUDED 1

#include <iostream>

#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/numpy/mpl/types_from_fctptr_signature.hpp>
#include <boost/numpy/dstream/wiring.hpp>
#include <boost/numpy/dstream/wiring/arg_from_core_shape_data.hpp>
#include <boost/numpy/dstream/wiring/arg_type_to_array_dtype.hpp>
#include <boost/numpy/dstream/wiring/return_type_to_array_dtype.hpp>

namespace ndhist {

namespace detail {

template <class MappingDefinition, class FTypes>
struct fill_wiring_model_api
{
    template <unsigned Idx>
    struct out_arr_value_type
    {
        typedef typename FTypes::return_type
                return_type;
        typedef typename boost::numpy::dstream::wiring::converter::detail::return_type_to_array_dtype<typename MappingDefinition::out, return_type, Idx>::type
                type;
    };

    template <unsigned Idx>
    struct in_arr_value_type
    {
        typedef typename boost::numpy::mpl::fct_arg_type<FTypes, Idx>::type
                arg_type;
        typedef typename boost::numpy::dstream::wiring::converter::detail::arg_type_to_array_dtype<arg_type>::type
                type;
    };

    template <unsigned Idx>
    struct out_arr_iter_operand_flags
    {
        // This should be appropriate for 99% of all cases. But one might have
        // this as a general "flag selector" interface.
        typedef boost::mpl::bitor_<
                    typename boost::numpy::detail::iter_operand::flags::WRITEONLY
                  , typename boost::numpy::detail::iter_operand::flags::NBO
                  , typename boost::numpy::detail::iter_operand::flags::ALIGNED
                >
                type;
    };

    template <unsigned Idx>
    struct in_arr_iter_operand_flags
    {
        // This should be appropriate for 99% of all cases. But one might have
        // this as a general "flag selector" interface.
        typedef typename boost::numpy::detail::iter_operand::flags::READONLY
                type;
    };

    template <class LoopService>
    struct iter_flags
    {
        // If any array data type is boost::python::object, the REF_OK iterator
        // operand flag needs to be set.
        // Note: This could lead to the requirement that
        //       the python GIL cannot released during the iteration!
        typedef typename boost::mpl::if_<
                  typename LoopService::object_arrays_are_involved
                , boost::numpy::detail::iter::flags::REFS_OK
                , boost::numpy::detail::iter::flags::NONE
                >::type
                refs_ok_flag;

        typedef boost::mpl::bitor_<
                    typename boost::numpy::detail::iter::flags::DONT_NEGATE_STRIDES
                  , refs_ok_flag
                >
                type;
    };

    BOOST_STATIC_CONSTANT(boost::numpy::order_t, order = boost::numpy::KEEPORDER);

    BOOST_STATIC_CONSTANT(boost::numpy::casting_t, casting = boost::numpy::SAME_KIND_CASTING);

    BOOST_STATIC_CONSTANT(intptr_t, buffersize = 0);
};

}// namespace detail

template <
    class MappingDefinition
  , class FTypes
>
struct fill_wiring_model
{
    typedef fill_wiring_model<MappingDefinition, FTypes>
            type;

    typedef detail::fill_wiring_model_api<MappingDefinition, FTypes>
            api;

    #define NDHIST_DEF(z, n, data) \
        typedef typename api::template in_arr_value_type<n>::type \
                BOOST_PP_CAT(in_arr_data_holding_t,n);
    BOOST_PP_REPEAT(1, NDHIST_DEF, ~)
    #undef NDHIST_DEF

    template <class ClassT, class FCaller>
    static
    void
    iterate(
          ClassT & self
        , FCaller const & f_caller
        , boost::numpy::detail::iter & iter
        , std::vector< std::vector<intptr_t> > const & core_shapes
        , bool & error_flag
    )
    {
        // Define the arg_from_core_shape_data converter types for all input
        // arguments.
        typedef typename boost::numpy::dstream::wiring::converter::detail::arg_from_core_shape_data_converter<
                    typename boost::numpy::mpl::fct_arg_type<FTypes, 0>::type
                  , in_arr_data_holding_t0
                >::type
                arg_converter_t0;

        do {
            intptr_t size = iter.get_inner_loop_size();
            while(size--)
            {
                std::cout << "Calling f:";
                f_caller.call(
                    self
                  , arg_converter_t0::apply(iter, MappingDefinition::out::arity + 0, core_shapes[MappingDefinition::out::arity + 0])
                );

                iter.add_inner_loop_strides_to_data_ptrs();
            }
        } while(iter.next());
    }

};

struct fill_wiring_model_selector
  : boost::numpy::dstream::wiring::wiring_model_selector_type
{
    typedef fill_wiring_model_selector
            type;

    template <
         class MappingDefinition
       , class FTypes
    >
    struct select
    {
        typedef fill_wiring_model<MappingDefinition, FTypes>
                type;
    };
};


}// namespace ndhist

#endif // !NDHIST_DETAIL_FILL_WIRING_MODEL_HPP_INCLUDED
