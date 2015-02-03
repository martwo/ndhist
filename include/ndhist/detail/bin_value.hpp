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
struct bin_value<bp::object>
{
    uintptr_t  * noe_;

    uintptr_t  * sow_obj_ptr_;
    bp::object   sow_obj_;
    bp::object * sow_;

    uintptr_t  * sows_obj_ptr_;
    bp::object   sows_obj_;
    bp::object * sows_;
};

}//namespace detail
}//namespace ndhist

#endif // !NDHIST_DETAIL_BIN_VALUE_HPP_INCLUDED
