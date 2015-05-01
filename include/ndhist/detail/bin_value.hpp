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
#ifndef NDHIST_DETAIL_BIN_VALUE_HPP_INCLUDED
#define NDHIST_DETAIL_BIN_VALUE_HPP_INCLUDED

#include <boost/python.hpp>

namespace ndhist {
namespace detail {

template <typename WeightValueType>
struct bin_value
{
    uintptr_t       * noe_;

    WeightValueType * sow_;

    WeightValueType * sows_;
};

template <>
struct bin_value<boost::python::object>
{
    uintptr_t             * noe_;

    uintptr_t             * sow_obj_ptr_;
    boost::python::object   sow_obj_;
    boost::python::object * sow_;

    uintptr_t             * sows_obj_ptr_;
    boost::python::object   sows_obj_;
    boost::python::object * sows_;
};

}//namespace detail
}//namespace ndhist

#endif // !NDHIST_DETAIL_BIN_VALUE_HPP_INCLUDED
