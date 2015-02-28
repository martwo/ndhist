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
#ifndef NDHIST_DETAIL_BIN_ITER_VALUE_TYPE_TRAITS_HPP_INCLUDED
#define NDHIST_DETAIL_BIN_ITER_VALUE_TYPE_TRAITS_HPP_INCLUDED 1

#include <vector>

#include <boost/python.hpp>

#include <boost/numpy.hpp>
#include <boost/numpy/iterators/value_type_traits.hpp>

#include <ndhist/detail/bin_value.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace detail {

template <typename WeightValueType>
struct bin_iter_value_type_traits
  : bn::iterators::value_type_traits
{
    typedef bin_iter_value_type_traits<WeightValueType>
            type_t;

    typedef bin_value<WeightValueType>
            value_type;
    typedef value_type &
            value_ref_type;
    typedef value_type *
            value_ptr_type;

    bin_iter_value_type_traits()
    {}

    bin_iter_value_type_traits(bn::ndarray const & arr)
      : fields_byte_offsets_(arr.get_dtype().get_fields_byte_offsets())
    {}

    std::vector<intptr_t> fields_byte_offsets_;
    value_type bin_value_;

    static
    void
    set_value(
        bn::iterators::value_type_traits & vtt_base
      , char * data_ptr
      , value_ref_type newbin
    )
    {
        value_ref_type bin = type_t::dereference(vtt_base, data_ptr);

        *bin.noe_  = *newbin.noe_;
        *bin.sow_  = *newbin.sow_;
        *bin.sows_ = *newbin.sows_;
    }

    static
    value_ref_type
    dereference(
        bn::iterators::value_type_traits & vtt_base
      , char * data_ptr
    )
    {
        type_t & vtt = *static_cast<type_t *>(&vtt_base);

        vtt.bin_value_.noe_  = reinterpret_cast<uintptr_t *>(data_ptr);
        vtt.bin_value_.sow_  = reinterpret_cast<WeightValueType *>(data_ptr + vtt.fields_byte_offsets_[1]);
        vtt.bin_value_.sows_ = reinterpret_cast<WeightValueType *>(data_ptr + vtt.fields_byte_offsets_[2]);

        return vtt.bin_value_;
    }
};

template <>
struct bin_iter_value_type_traits<bp::object>
  : bn::iterators::value_type_traits
{
    typedef bin_iter_value_type_traits<bp::object>
            type_t;

    typedef bin_value<bp::object>
            value_type;
    typedef value_type &
            value_ref_type;
    typedef value_type *
            value_ptr_type;

    bin_iter_value_type_traits()
    {}

    bin_iter_value_type_traits(bn::ndarray const & arr)
      : fields_byte_offsets_(arr.get_dtype().get_fields_byte_offsets())
    {}

    std::vector<intptr_t> fields_byte_offsets_;
    value_type bin_value_;

    static
    void
    set_value(
        bn::iterators::value_type_traits & vtt_base
      , char * data_ptr
      , value_ref_type newbin
    )
    {
        value_ref_type bin = type_t::dereference(vtt_base, data_ptr);

        *bin.noe_          = *newbin.noe_;
        *bin.sow_obj_ptr_  = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(newbin.sow_obj_.ptr()));
        *bin.sows_obj_ptr_ = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(newbin.sows_obj_.ptr()));
    }

    static
    value_ref_type
    dereference(
        bn::iterators::value_type_traits & vtt_base
      , char * data_ptr
    )
    {
        type_t & vtt = *static_cast<type_t *>(&vtt_base);

        vtt.bin_value_.noe_ = reinterpret_cast<uintptr_t*>(data_ptr);

        vtt.bin_value_.sow_obj_ptr_ = reinterpret_cast<uintptr_t*>(data_ptr + vtt.fields_byte_offsets_[1]);
        if(*(vtt.bin_value_.sow_obj_ptr_) == 0) {
            vtt.bin_value_.sow_obj_ = bp::object();
        }
        else {
            vtt.bin_value_.sow_obj_ = bp::object(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*vtt.bin_value_.sow_obj_ptr_)));
        }
        vtt.bin_value_.sow_ = &vtt.bin_value_.sow_obj_;

        vtt.bin_value_.sows_obj_ptr_ = reinterpret_cast<uintptr_t*>(data_ptr + vtt.fields_byte_offsets_[2]);
        if(*(vtt.bin_value_.sows_obj_ptr_) == 0) {
            vtt.bin_value_.sows_obj_ = bp::object();
        }
        else {
            vtt.bin_value_.sows_obj_ = bp::object(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*vtt.bin_value_.sows_obj_ptr_)));
        }
        vtt.bin_value_.sows_ = &vtt.bin_value_.sows_obj_;

        return vtt.bin_value_;
    }
};

}//namespace detail
}//namespace ndhist

#endif // !NDHIST_DETAIL_BIN_ITER_VALUE_TYPE_TRAITS_HPP_INCLUDED
