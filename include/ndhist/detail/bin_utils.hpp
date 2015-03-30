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
#ifndef NDHIST_DETAIL_BIN_UTILS_HPP_INCLUDED
#define NDHIST_DETAIL_BIN_UTILS_HPP_INCLUDED 1

#include <ndhist/detail/bin_value.hpp>

namespace ndhist {
namespace detail {

// template <typename WeightValueType>
// struct bin_utils;
//
// template <typename WeightValueType>
// struct bin_utils_base
// {
//     static
//     void
//     get_bin_by_indices(
//         ndhist const & self
//       , bin_value<WeightValueType> & bin
//       , std::vector<intptr_t> const & indices
//     )
//     {
//         uintptr_t const nd = self.get_nd();
//         char * data_addr = self.bc_.get_data() + self.bc_.get_bytearray_data_offset() + self.bc_.calc_first_shape_element_data_offset();
//         std::vector<intptr_t> const & strides = self.bc_.get_data_strides_vector();
//         for(uintptr_t i=0; i<nd; ++i)
//         {
//             data_addr += indices[i]*strides[i];
//         }
//         bin_utils<WeightValueType>::get_bin(bin, data_addr);
//     }
// };

template <typename WeightValueType>
struct bin_utils
//  : bin_utils_base<WeightValueType>
{
    typedef WeightValueType &
            weight_ref_type;

    static
    weight_ref_type
    get_weight_type_value_from_iter(bn::detail::iter & iter, int op_idx)
    {
        return *reinterpret_cast<WeightValueType*>(iter.get_data(op_idx));
    }

    static
    void
    increment_bin(char * bc_data_addr, WeightValueType const & weight)
    {
        uintptr_t       & noe  = *reinterpret_cast<uintptr_t*>(bc_data_addr);
        WeightValueType & sow  = *reinterpret_cast<WeightValueType*>(bc_data_addr + sizeof(uintptr_t));
        WeightValueType & sows = *reinterpret_cast<WeightValueType*>(bc_data_addr + sizeof(uintptr_t) + sizeof(WeightValueType));

        noe  += 1;
        sow  += weight;
        sows += weight * weight;
    }

    static
    void
    get_bin(bin_value<WeightValueType> & bin, char * data_addr)
    {
        bin.noe_  = reinterpret_cast<uintptr_t*>(data_addr);
        bin.sow_  = reinterpret_cast<WeightValueType*>(data_addr + sizeof(uintptr_t));
        bin.sows_ = reinterpret_cast<WeightValueType*>(data_addr + sizeof(uintptr_t) + sizeof(WeightValueType));
    }

    static
    void
    set_value_from_data(char * dst_addr, char * src_addr)
    {
        WeightValueType & dst_value = *reinterpret_cast<WeightValueType*>(dst_addr);
        WeightValueType & src_value = *reinterpret_cast<WeightValueType*>(src_addr);

        dst_value = src_value;
    }

    static
    void
    zero_bin(char * bc_data_addr)
    {
        uintptr_t       & noe  = *reinterpret_cast<uintptr_t*>(bc_data_addr);
        WeightValueType & sow  = *reinterpret_cast<WeightValueType*>(bc_data_addr + sizeof(uintptr_t));
        WeightValueType & sows = *reinterpret_cast<WeightValueType*>(bc_data_addr + sizeof(uintptr_t) + sizeof(WeightValueType));

        noe  = 0;
        sow  = WeightValueType(0);
        sows = WeightValueType(0);
    }
};

template <>
struct bin_utils<bp::object>
//  : bin_utils_base<bp::object>
{
    typedef bp::object
            weight_ref_type;

    static
    weight_ref_type
    get_weight_type_value_from_iter(bn::detail::iter & iter, int op_idx)
    {
        uintptr_t * value_ptr = reinterpret_cast<uintptr_t*>(iter.get_data(op_idx));
        bp::object value(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*value_ptr)));
        return value;
    }

    static
    void
    increment_bin(char * bc_data_addr, bp::object const & weight)
    {
        uintptr_t & noe = *reinterpret_cast<uintptr_t*>(bc_data_addr);
        uintptr_t * ptr = reinterpret_cast<uintptr_t*>(bc_data_addr + sizeof(uintptr_t));
        bp::object sow(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*ptr)));
        uintptr_t * ptr2 = reinterpret_cast<uintptr_t*>(bc_data_addr + 2*sizeof(uintptr_t));
        bp::object sows(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*ptr2)));

        noe  += 1;
        sow  += weight;
        sows += weight * weight;
    }

    static
    void
    get_bin(bin_value<bp::object> & bin, char * data_addr)
    {
        bin.noe_  = reinterpret_cast<uintptr_t*>(data_addr);

        bin.sow_obj_ptr_ = reinterpret_cast<uintptr_t*>(data_addr + sizeof(uintptr_t));
        bin.sow_obj_ = bp::object(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*bin.sow_obj_ptr_)));
        bin.sow_ = &bin.sow_obj_;

        bin.sows_obj_ptr_ = reinterpret_cast<uintptr_t*>(data_addr + 2*sizeof(uintptr_t));
        bin.sows_obj_ = bp::object(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*bin.sows_obj_ptr_)));
        bin.sows_ = &bin.sows_obj_;
    }

    static
    void
    set_value_from_data(char * dst_addr, char * src_addr)
    {
        uintptr_t * dst_obj_ptr_ptr = reinterpret_cast<uintptr_t *>(dst_addr);
        uintptr_t * src_obj_ptr_ptr = reinterpret_cast<uintptr_t *>(src_addr);
        bp::object src_obj(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*src_obj_ptr_ptr)));
        bp::xdecref<PyObject>(reinterpret_cast<PyObject*>(*dst_obj_ptr_ptr));
        *dst_obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(src_obj.ptr()));
    }

    static
    void
    zero_bin(char * bc_data_addr)
    {
        uintptr_t & noe = *reinterpret_cast<uintptr_t*>(bc_data_addr);
        noe = 0;

        bp::object sow_obj = bp::object(0);
        uintptr_t * ptr = reinterpret_cast<uintptr_t*>(bc_data_addr + sizeof(uintptr_t));
        bp::xdecref<PyObject>(reinterpret_cast<PyObject*>(*ptr));
        *ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sow_obj.ptr()));

        bp::object sows_obj = bp::object(0);
        uintptr_t * ptr2 = reinterpret_cast<uintptr_t*>(bc_data_addr + 2*sizeof(uintptr_t));
        bp::xdecref<PyObject>(reinterpret_cast<PyObject*>(*ptr2));
        *ptr2 = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sows_obj.ptr()));
    }
};

}//namespace detail
}//namespace ndhist

#endif // !NDHIST_DETAIL_BIN_UTILS_HPP_INCLUDED
