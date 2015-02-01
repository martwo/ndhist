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
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <sstream>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/python/list.hpp>
#include <boost/python/refcount.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/iterators/flat_iterator.hpp>
#include <boost/numpy/iterators/indexed_iterator.hpp>
#include <boost/numpy/iterators/multi_flat_iterator.hpp>
#include <boost/numpy/dstream.hpp>
#include <boost/numpy/utilities.hpp>

#include <ndhist/limits.hpp>
#include <ndhist/ndhist.hpp>
#include <ndhist/axis.hpp>
#include <ndhist/detail/axis_index_iter.hpp>
#include <ndhist/detail/limits.hpp>
//#include <ndhist/detail/full_multi_axis_index_iter.hpp>
#include <ndhist/detail/utils.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

namespace detail {

static
uintptr_t
calc_n_oor_bins(std::vector<intptr_t> const & shape)
{
    uintptr_t n = 1;
    uintptr_t n_shape = 1;
    for(uintptr_t i=0; i<shape.size(); ++i)
    {
        n_shape *= shape[i];
        n *= shape[i]+2;
    }
    n -= n_shape;
    return n;
}

template <typename BCValueType>
struct bin_value
{
    uintptr_t   * noe_;
    BCValueType * sow_;
    BCValueType * sows_;
};
template <>
struct bin_value<bp::object>
{
    uintptr_t  * noe_;
    uintptr_t  * sow_obj_ptr_;
    bp::object sow_obj_;
    bp::object * sow_;
    uintptr_t  * sows_obj_ptr_;
    bp::object sows_obj_;
    bp::object * sows_;
};

template <typename BCValueType>
struct bc_value_traits
{
    typedef BCValueType &
            ref_type;

    static
    ref_type
    get_value_from_iter(bn::detail::iter & iter, int op_idx)
    {
        return *reinterpret_cast<BCValueType*>(iter.get_data(op_idx));
    }

    static
    void
    increment_bin(char * bc_data_addr, BCValueType const & weight)
    {
        uintptr_t & noe    = *reinterpret_cast<uintptr_t*>(bc_data_addr);
        BCValueType & sow  = *reinterpret_cast<BCValueType*>(bc_data_addr + sizeof(uintptr_t));
        BCValueType & sows = *reinterpret_cast<BCValueType*>(bc_data_addr + sizeof(uintptr_t) + sizeof(BCValueType));

        noe  += 1;
        sow  += weight;
        sows += weight * weight;
    }

    static
    void
    get_bin(bin_value<BCValueType> & bin, char * data_addr)
    {
        bin.noe_  = reinterpret_cast<uintptr_t*>(data_addr);
        bin.sow_  = reinterpret_cast<BCValueType*>(data_addr + sizeof(uintptr_t));
        bin.sows_ = reinterpret_cast<BCValueType*>(data_addr + sizeof(uintptr_t) + sizeof(BCValueType));
    }

    static
    void
    set_value_from_data(char * dst_addr, char * src_addr)
    {
        BCValueType & dst_value = *reinterpret_cast<BCValueType*>(dst_addr);
        BCValueType & src_value = *reinterpret_cast<BCValueType*>(src_addr);

        dst_value = src_value;
    }
};

template <>
struct bc_value_traits<bp::object>
{
    typedef bp::object
            ref_type;

    static
    ref_type
    get_value_from_iter(bn::detail::iter & iter, int op_idx)
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
        *dst_obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(src_obj.ptr()));
    }
};


template <typename BCValueType>
struct bin_value_type_traits
  : bn::iterators::value_type_traits
{
    typedef bin_value_type_traits<BCValueType>
            type_t;

    typedef bin_value<BCValueType>
            value_type;
    typedef value_type &
            value_ref_type;
    typedef value_type *
            value_ptr_type;

    bin_value_type_traits()
    {}

    bin_value_type_traits(bn::ndarray const & arr)
      : fields_byte_offsets_(arr.get_dtype().get_fields_byte_offsets())
    {}

    std::vector<intptr_t> fields_byte_offsets_;
    bin_value<BCValueType> bin_value_;

    static
    value_ref_type
    dereference(
        bn::iterators::value_type_traits & vtt_base
      , char * data_ptr
    )
    {
        type_t & vtt = *static_cast<type_t *>(&vtt_base);

        vtt.bin_value_.noe_  = reinterpret_cast<uintptr_t *>(data_ptr);
        vtt.bin_value_.sow_  = reinterpret_cast<BCValueType *>(data_ptr + vtt.fields_byte_offsets_[1]);
        vtt.bin_value_.sows_ = reinterpret_cast<BCValueType *>(data_ptr + vtt.fields_byte_offsets_[2]);

        return vtt.bin_value_;
    }
};
template <>
struct bin_value_type_traits<bp::object>
  : bn::iterators::value_type_traits
{
    typedef bin_value_type_traits<bp::object>
            type_t;

    typedef bin_value<bp::object>
            value_type;
    typedef value_type &
            value_ref_type;
    typedef value_type *
            value_ptr_type;

    bin_value_type_traits()
    {}

    bin_value_type_traits(bn::ndarray const & arr)
      : fields_byte_offsets_(arr.get_dtype().get_fields_byte_offsets())
    {}

    std::vector<intptr_t> fields_byte_offsets_;
    bin_value<bp::object> bin_value_;

    static
    value_ref_type
    dereference(
        bn::iterators::value_type_traits & vtt_base
      , char * data_ptr
    )
    {
        type_t & vtt = *static_cast<type_t *>(&vtt_base);

        vtt.bin_value_.noe_  = reinterpret_cast<uintptr_t*>(data_ptr);

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

template <typename BCValueType>
static
void
get_noor_bin(
    ndhist const & self
  , bin_value<BCValueType> & bin
  , std::vector<intptr_t> const & indices
)
{
    uintptr_t const nd = self.get_nd();
    char * data_addr = self.bc_->get_data() + self.bc_->calc_data_offset(0);
    std::vector<intptr_t> const & strides = self.bc_->get_data_strides_vector();
    for(uintptr_t i=0; i<nd; ++i)
    {
        data_addr += indices[i]*strides[i];
    }
    bc_value_traits<BCValueType>::get_bin(bin, data_addr);
}

// template <typename BCValueType>
// static
// void
// flush_oor_cache(
//     ndhist                          & self
//   , OORFillRecordStack<BCValueType> & oorfrstack
//   , std::vector<intptr_t> const     & f_n_extra_bins_vec
//   , uintptr_t const                   bc_data_offset
// )
// {
//     typedef typename bc_value_traits<BCValueType>::ref_type
//             bc_ref_type;
//
//     intptr_t const nd = f_n_extra_bins_vec.size();
//
//     // Fill in the cached values.
//     std::vector<intptr_t> const * arr_strides_ptr;
//     char * bin_data_addr;
//     intptr_t idx = oorfrstack.get_size();
//     while(idx--)
//     {
//         typename OORFillRecordStack<BCValueType>::oor_fill_record_type const & rec = oorfrstack.get_record(idx);
//         arr_strides_ptr = NULL;
//         bin_data_addr = NULL;
//         if(rec.is_oor)
//         {
//             boost::shared_ptr<detail::ndarray_storage> & oor_arr_storage = self.oor_arr_vec_[rec.oor_arr_idx];
//
//             arr_strides_ptr = &oor_arr_storage->get_data_strides_vector();
//             bin_data_addr = oor_arr_storage->data_ + oor_arr_storage->CalcDataOffset(0);
//
//         }
//         else
//         {
//             arr_strides_ptr = &self.bc_->get_data_strides_vector();
//             bin_data_addr = self.bc_->data_ + bc_data_offset;
//         }
//
//         // Translate the relative indices into an absolute
//         // data address for the extended bin content array.
//         for(intptr_t axis=0; axis<nd; ++axis)
//         {
//             bin_data_addr += (f_n_extra_bins_vec[axis] + rec.relative_indices[axis]) * (*arr_strides_ptr)[axis];
//         }
//
//         detail::bc_value_traits<BCValueType>::increment_bin(bin_data_addr, rec.weight);
//     }
//
//     // Finally, clear the stack.
//     oorfrstack.clear();
// }

template <typename BCValueType>
struct iadd_fct_traits
{
    static
    void apply(ndhist & self, ndhist const & other)
    {
        if(! self.is_compatible(other))
        {
            std::stringstream ss;
            ss << "The += operator is only defined for two compatible ndhist "
               << "objects!";
            throw AssertionError(ss.str());
        }

        // Add the bin contents of the two ndhist objects.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_value_type_traits<BCValueType>
                  , bin_value_type_traits<BCValueType>
                >
                multi_iter_t;

        bp::object data_owner;
        bn::ndarray self_bc_arr = self.bc_->construct_ndarray(self.bc_->get_dtype(), 0, &data_owner);
        bn::ndarray other_bc_arr = other.bc_->construct_ndarray(other.bc_->get_dtype(), 0, &data_owner);
        multi_iter_t bc_it(
            self_bc_arr
          , other_bc_arr
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value);
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value  = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 other_bin_value = multi_value.value_1;
            *self_bin_value.noe_  += *other_bin_value.noe_;
            *self_bin_value.sow_  += *other_bin_value.sow_;
            *self_bin_value.sows_ += *other_bin_value.sows_;
            ++bc_it;
        }
    }
};

template <typename BCValueType>
struct idiv_fct_traits
{
    static
    void apply(ndhist & self, bn::ndarray const & value_arr)
    {
        // Divide the bin contents of the ndhist object with the
        // scalar value.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_value_type_traits<BCValueType>
                  , bn::iterators::single_value<BCValueType>
                >
                multi_iter_t;

        bp::object data_owner;
        bn::ndarray self_bc_arr = self.bc_->construct_ndarray(self.bc_->get_dtype(), 0, &data_owner);
        multi_iter_t bc_it(
            self_bc_arr
          , const_cast<bn::ndarray &>(value_arr)
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value
        );
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 value          = multi_value.value_1;
            *self_bin_value.sow_  /= value;
            *self_bin_value.sows_ /= value * value;
            ++bc_it;
        }
    }
};

template <typename BCValueType>
struct imul_fct_traits
{
    static
    void apply(ndhist & self, bn::ndarray const & value_arr)
    {
        // Multiply the bin contents of the ndhist object with the
        // scalar value.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_value_type_traits<BCValueType>
                  , bn::iterators::single_value<BCValueType>
                >
                multi_iter_t;

        bp::object data_owner;
        bn::ndarray self_bc_arr = self.bc_->construct_ndarray(self.bc_->get_dtype(), 0, &data_owner);
        multi_iter_t bc_it(
            self_bc_arr
          , const_cast<bn::ndarray &>(value_arr)
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value
        );
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 value          = multi_value.value_1;
            *self_bin_value.sow_  *= value;
            *self_bin_value.sows_ *= value * value;
            ++bc_it;
        }
    }
};


// template <typename BCValueType>
// struct project_fct_traits
// {
//     static
//     ndhist
//     apply(ndhist const & self, std::set<intptr_t> const & axes)
//     {
//         // Create a ndhist with the dimensions specified by axes.
//         uintptr_t const self_nd = self.get_nd();
//         uintptr_t const proj_nd = axes.size();
//
//         bp::list axis_list;
//         std::set<intptr_t>::const_iterator axes_it = axes.begin();
//         std::set<intptr_t>::const_iterator axes_end = axes.end();
//         for(; axes_it != axes_end; ++axes_it)
//         {
//             axis_list.append(self.get_axis_definition(*axes_it));
//         }
//         bp::tuple axes_tuple(axis_list);
//         ndhist proj(axes_tuple, self.bc_weight_dt_, self.bc_class_);
//
//         full_multi_axis_index_iter proj_idx_iter(proj.bc_->get_shape_vector());
//         full_multi_axis_index_iter self_idx_iter(self.bc_->get_shape_vector());
//
//         // Iterate over *all* the bins (including OOR bins) of the projection.
//         std::vector<intptr_t> proj_fixed_axes_indices(proj_nd, axis::FLAGS_FLOATING_INDEX);
//         std::vector<intptr_t> self_fixed_axes_indices(self_nd, axis::FLAGS_FLOATING_INDEX);
//         bin_value<BCValueType> proj_bin;
//         bin_value<BCValueType> self_bin;
//         proj_idx_iter.init_iteration(proj_fixed_axes_indices);
//         while(! proj_idx_iter.is_end())
//         {
//             std::vector<intptr_t> const & proj_indices = proj_idx_iter.get_indices();
//             // Get the proj bin.
//             if(proj_idx_iter.is_oor_bin()) {
//                 get_oor_bin(proj, proj_bin, proj_idx_iter.get_oor_array_index(), proj_indices);
//             }
//             else {
//                 get_noor_bin(proj, proj_bin, proj_indices);
//             }
//
//             // Iterate over all the axes of self which are not fixed through the
//             // current projection indices.
//             axes_it = axes.begin();
//             for(uintptr_t i=0; axes_it != axes_end; ++axes_it, ++i)
//             {
//                 self_fixed_axes_indices[*axes_it] = proj_indices[i];
//             }
//
//             self_idx_iter.init_iteration(self_fixed_axes_indices);
//             while(! self_idx_iter.is_end())
//             {
//                 std::vector<intptr_t> const & self_indices = self_idx_iter.get_indices();
//                 // Get the self bin.
//                 if(self_idx_iter.is_oor_bin()) {
//                     get_oor_bin(self, self_bin, self_idx_iter.get_oor_array_index(), self_indices);
//                 }
//                 else {
//                     get_noor_bin(self, self_bin, self_indices);
//                 }
//
//                 // Add the self bin to the proj bin.
//                 *proj_bin.noe_  += *self_bin.noe_;
//                 *proj_bin.sow_  += *self_bin.sow_;
//                 *proj_bin.sows_ += *self_bin.sows_;
//
//                 self_idx_iter.increment();
//             }
//
//             proj_idx_iter.increment();
//         }
//
//         return proj;
//     }
// };

// template <typename BCValueType>
// std::vector<bn::ndarray>
// get_field_axes_oor_ndarrays(
//     ndhist const & self
//   , ::ndhist::axis::out_of_range_t const oortype
//   , size_t const field_idx
// )
// {
//     uintptr_t const nd = self.get_nd();
//     std::vector<bn::ndarray> array_vec;
//     array_vec.reserve(nd);
//     uintptr_t oor_arr_idx;
//     std::vector<intptr_t> indices(nd);
//     std::vector<intptr_t> oor_arr_indices(nd);
//     std::vector<intptr_t> shape(nd);
//     std::vector<intptr_t> const fields_byte_offsets = self.bc_->get_dtype().get_fields_byte_offsets();
//     char * oor_data_addr;
//     for(uintptr_t i=0; i<nd; ++i)
//     {
//         std::vector<intptr_t> const & histshape = self.bc_->get_shape_vector();
//         for(uintptr_t j=0; j<nd; ++j)
//         {
//             shape[j] = (j == i ? 1 : histshape[j] + 2);
//         }
//
//         bn::ndarray arr = bn::empty(shape, (field_idx == 0 ? self.bc_noe_dt_ : self.bc_weight_dt_));
//         bn::iterators::indexed_iterator< bn::iterators::single_value<BCValueType> > iter(arr, bn::detail::iter_operand::flags::READWRITE::value);
//         while(! iter.is_end())
//         {
//             iter.get_indices(indices);
//
//             // Determine the oor array index and the data address of the bin.
//             oor_arr_idx = 0;
//             for(uintptr_t j=0; j<nd; ++j)
//             {
//                 // Mark if the axis j is available.
//                 if(j != i)
//                 {
//                     if(indices[j] == 0)
//                     {
//                         // It's the underflow bin of axis j.
//                         oor_arr_indices[j] = 0;
//                     }
//                     else if(indices[j] == histshape[j]+1) // Notice the underflow element.
//                     {
//                         // It's the overflow bin of axis j.
//                         oor_arr_indices[j] = 1;
//                     }
//                     else
//                     {
//                         // It's a normal bin of axis j.
//                         oor_arr_idx |= (1<<j);
//                         oor_arr_indices[j] = indices[j] - 1; // Notice the underflow element.
//                     }
//                 }
//             }
//             // Set the index of the out-of-range axis of interest.
//             oor_arr_indices[i] = (oortype == axis::OOR_UNDERFLOW ? 0 : 1);
//
//             boost::shared_ptr<detail::ndarray_storage> const & oor_arr_storage = self.oor_arr_vec_[oor_arr_idx];
//             std::vector<intptr_t> const & oor_arr_strides = oor_arr_storage->get_data_strides_vector();
//             oor_data_addr = oor_arr_storage->data_ + oor_arr_storage->CalcDataOffset(fields_byte_offsets[field_idx]);
//             for(size_t j=0; j<nd; ++j)
//             {
//                 oor_data_addr += oor_arr_indices[j]*oor_arr_strides[j];
//             }
//
//             // Set the element of the new ndarray to the (sub) element of the
//             // oor array.
//             bc_value_traits<BCValueType>::set_value_from_data(iter.get_detail_iter().get_data(0), oor_data_addr);
//
//             ++iter;
//         }
//
//         array_vec.push_back(arr);
//     }
//
//     return array_vec;
// }

/*
struct generic_nd_traits
{
    template <typename BCValueType>
    struct fill_traits
    {
        static
        void
        fill(ndhist & self, bp::object const & ndvalues_obj, bp::object const & weight_obj)
        {
            // The ndvalues_obj object is supposed to be a structured ndarray.
            // But in case of 1-dimensional histogram we accept also a
            // simple array, which can also be a normal Python list.

            size_t const nd = self.get_nd();
            bp::object ndvalues_arr_obj;
            try
            {
                ndvalues_arr_obj = bn::from_object(ndvalues_obj, self.get_ndvalues_dtype(), 0, 0, bn::ndarray::ALIGNED);
            }
            catch (const bp::error_already_set &)
            {
                if(nd == 1)
                {
                    ndvalues_arr_obj = bn::from_object(ndvalues_obj, self.get_axes()[0]->get_dtype(), 0, 0, bn::ndarray::ALIGNED);
                }
                else
                {
                    std::stringstream ss;
                    ss << "The ndvalues parameter must either be a tuple of "
                       << nd << " one-dimensional arrays or one structured "
                       << "ndarray!";
                    throw TypeError(ss.str());
                }
            }
            bn::ndarray ndvalue_arr = *static_cast<bn::ndarray*>(&ndvalues_arr_obj);
            bn::ndarray weight_arr = bn::from_object(weight_obj, bn::dtype::get_builtin<BCValueType>(), 0, 0, bn::ndarray::ALIGNED);

            // Construct an iterator for the input arrays. We use the loop service
            // of BoostNumpy that determines the number of loop
            // dimensions automatically and provides generalized universal
            // functions.
            typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
                    core_shape_t0;
            typedef bn::dstream::array_definition< core_shape_t0, void>
                    in_arr_def0;
            typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
                    core_shape_t1;
            typedef bn::dstream::array_definition< core_shape_t1, BCValueType>
                    in_arr_def1;
            typedef bn::dstream::detail::loop_service_arity<2>::loop_service<in_arr_def0, in_arr_def1>
                    loop_service_t;

            bn::dstream::detail::input_array_service<in_arr_def0> in_arr_service0(ndvalue_arr);
            bn::dstream::detail::input_array_service<in_arr_def1> in_arr_service1(weight_arr);
            loop_service_t loop_service(in_arr_service0, in_arr_service1);

            bn::detail::iter_operand_flags_t in_arr_iter_op_flags0 = bn::detail::iter_operand::flags::READONLY::value;
            bn::detail::iter_operand_flags_t in_arr_iter_op_flags1 = bn::detail::iter_operand::flags::READONLY::value;

            bn::detail::iter_operand in_arr_iter_op0( in_arr_service0.get_arr(), in_arr_iter_op_flags0, in_arr_service0.get_arr_bcr_data() );
            bn::detail::iter_operand in_arr_iter_op1( in_arr_service1.get_arr(), in_arr_iter_op_flags1, in_arr_service1.get_arr_bcr_data() );

            bn::detail::iter_flags_t iter_flags =
                bn::detail::iter::flags::REFS_OK::value // This is needed for the
                                                        // weight, which can be bp::object.
              | bn::detail::iter::flags::EXTERNAL_LOOP::value;
            bn::order_t order = bn::KEEPORDER;
            bn::casting_t casting = bn::NO_CASTING;
            intptr_t buffersize = 0;

            bn::detail::iter iter(
                  iter_flags
                , order
                , casting
                , loop_service.get_loop_nd()
                , loop_service.get_loop_shape_data()
                , buffersize
                , in_arr_iter_op0
                , in_arr_iter_op1
            );
            iter.init_full_iteration();

            // Create an indexed iterator for the bin content array.
            bp::object self_obj(bp::ptr(&self));
            bn::ndarray bc_arr = self.bc_->ConstructNDArray(self.bc_weight_dt_, 1, &self_obj);
            bn::iterators::indexed_iterator< bn::iterators::single_value<BCValueType> > bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);

            // Get the byte offsets of the fields and check if the number of fields
            // match the dimensionality of the histogram.
            std::vector<intptr_t> ndvalue_byte_offsets = ndvalue_arr.get_dtype().get_fields_byte_offsets();

            if(ndvalue_byte_offsets.size() == 0)
            {
                if(nd == 1)
                {
                    ndvalue_byte_offsets.push_back(0);
                }
                else
                {
                    std::stringstream ss;
                    ss << "The dimensionality of the histogram is " << nd
                       << ", i.e. greater than 1, so the value ndarray must be a "
                       << "structured ndarray!";
                    throw ValueError(ss.str());
                }
            }
            else if(ndvalue_byte_offsets.size() != nd)
            {
                std::stringstream ss;
                ss << "The value ndarray must contain " << nd << " fields, one for "
                   << "each dimension! Right now it has "
                   << ndvalue_byte_offsets.size() << " fields!";
                throw ValueError(ss.str());
            }

            // Do the iteration.
            typedef typename bc_value_traits<BCValueType>::ref_type
                    bc_ref_type;
            std::vector<intptr_t> indices(nd);
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Get the weight scalar from the iterator.
                    bc_ref_type weight = bc_value_traits<BCValueType>::get_value_from_iter(iter, 1);

                    // Fill the scalar into the bin content array.
                    // Get the coordinate of the current ndvalue.
                    bool is_overflown = false;
                    for(size_t i=0; i<nd; ++i)
                    {
                        //std::cout << "Get bin idx of axis " << i << " of " << nd << std::endl;
                        boost::shared_ptr<detail::Axis> & axis = self.get_axes()[i];
                        char * ndvalue_ptr = iter.get_data(0) + ndvalue_byte_offsets[i];
                        axis::out_of_range_t oor;
                        intptr_t axis_idx = axis->get_bin_index_fct(axis->data_, ndvalue_ptr, &oor);
                        if(oor == axis::OOR_NONE)
                        {
                            indices[i] = axis_idx;
                        }
                        else
                        {
                            // The current value is out of the axis bounds.
                            // Just ignore it for now.
                            // TODO: Introduce an under- and overflow bin for each
                            //       each axis. Or resize the axis.
                            is_overflown = true;
                            break;
                        }
                    }

                    // Increase the bin content if the bin exists.
                    if(!is_overflown)
                    {
                        bc_iter.jump_to(indices);
                        bc_ref_type bc_value = *bc_iter;
                        bc_value += weight;
                    }

                    // Jump to the next fill iteration.
                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    }; // struct fill_traits
}; // struct generic_nd_traits
*/

template <int nd>
struct nd_traits;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND, <ndhist/ndhist.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail

ndhist::
ndhist(
    bp::tuple const & axes
  , bp::object const & dt
  , bp::object const & bc_class
)
  : nd_(bp::len(axes))
  , ndvalues_dt_(bn::dtype::new_builtin<void>())
  , bc_noe_dt_(bn::dtype::get_builtin<uintptr_t>())
  , bc_weight_dt_(bn::dtype(dt))
  , bc_class_(bc_class)
{
    std::vector<intptr_t> shape(nd_);
    axes_extension_max_fcap_vec_.resize(nd_);
    axes_extension_max_bcap_vec_.resize(nd_);

    // Get the axes of the histogram.
    for(size_t i=0; i<nd_; ++i)
    {
        boost::shared_ptr<Axis> axis = bp::extract< boost::shared_ptr<Axis> >(axes[i]);

        // Set an axis name if it is not specified.
        std::string & axis_name = axis->get_name();
        if(axis_name == "")
        {
            std::stringstream ss_axis_name;
            ss_axis_name << "a" << i;
            axis_name = ss_axis_name.str();
        }

        // Add the axis field to the ndvalues dtype object.
        ndvalues_dt_.add_field(axis_name, axis->get_dtype());

        // Add the bin content shape information for this axis. The number of
        // of bins of an axis include possible under- and overflow bins.
        shape[i] = axis->get_n_bins();

        // Set the extra front and back capacity for this axis if the axis is
        // extendable.
        if(axis->is_extendable())
        {
            axes_extension_max_fcap_vec_[i] = axis->get_extension_max_fcap();
            axes_extension_max_bcap_vec_[i] = axis->get_extension_max_bcap();
        }
        else
        {
            axes_extension_max_fcap_vec_[i] = 0;
            axes_extension_max_bcap_vec_[i] = 0;
        }

        // Add the axis to the axes vector.
        axes_.push_back(axis);
    }

    // TODO: Make this as an option in the constructor.
    intptr_t oor_stack_size = 65536;

    // Create a ndarray for the bin content. Each bin content element consists
    // of three sub-elements: n_entries, sum_of_weights, sum_of_weights_squared.
    bn::dtype bc_dt = bn::dtype::new_builtin<void>();
    bc_dt.add_field("n",    bc_noe_dt_);
    bc_dt.add_field("sow",  bc_weight_dt_);
    bc_dt.add_field("sows", bc_weight_dt_);
    bc_ = boost::shared_ptr<detail::ndarray_storage>(new detail::ndarray_storage(shape, axes_extension_max_fcap_vec_, axes_extension_max_bcap_vec_, bc_dt));

    // Set the fill function based on the bin content data type.
    bool bc_dtype_supported = false;
    #define BOOST_PP_ITERATION_PARAMS_1                                        \
        (4, (1, NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND, <ndhist/ndhist.hpp>, 2))
    #include BOOST_PP_ITERATE()
    else
    {
        // nd is greater than NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND.
        #define NDHIST_WEIGHT_VALUE_TYPE_SUPPORT(r, data, BCDTYPE)              \
            if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<BCDTYPE>()))\
            {                                                                   \
                if(bc_dtype_supported)                                          \
                {                                                               \
                    std::stringstream ss;                                       \
                    ss << "The bin content data type is supported by more than "\
                       << "one possible C++ data type! This is an internal "    \
                       << "error!";                                             \
                    throw TypeError(ss.str());                                  \
                }                                                               \
                oor_fill_record_stack_ = boost::shared_ptr< detail::OORFillRecordStack<BCDTYPE> >(new detail::OORFillRecordStack<BCDTYPE>(nd_, oor_stack_size));\
                /*fill_fct_ = &detail::generic_nd_traits::fill_traits<BCDTYPE>::fill;*/\
                bc_dtype_supported = true;                                      \
            }
        BOOST_PP_SEQ_FOR_EACH(NDHIST_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)
        #undef NDHIST_WEIGHT_VALUE_TYPE_SUPPORT
    }
    if(!bc_dtype_supported)
    {
        std::stringstream ss;
        ss << "The data type of the bin content array is not supported.";
        throw TypeError(ss.str());
    }

    // Setup the function pointers, which depend on the weight value type.
    #define NDHIST_WEIGHT_VALUE_TYPE_SUPPORT(r, data, WEIGHT_VALUE_TYPE)        \
        if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<WEIGHT_VALUE_TYPE>()))\
        {                                                                       \
            iadd_fct_ = &detail::iadd_fct_traits<WEIGHT_VALUE_TYPE>::apply;     \
            idiv_fct_ = &detail::idiv_fct_traits<WEIGHT_VALUE_TYPE>::apply;     \
            imul_fct_ = &detail::imul_fct_traits<WEIGHT_VALUE_TYPE>::apply;     \
            /*get_weight_type_field_axes_oor_ndarrays_fct_ = &detail::get_field_axes_oor_ndarrays<WEIGHT_VALUE_TYPE>;*/\
            /*project_fct_ = &detail::project_fct_traits<WEIGHT_VALUE_TYPE>::apply;*/\
        }
    BOOST_PP_SEQ_FOR_EACH(NDHIST_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)
    #undef NDHIST_WEIGHT_VALUE_TYPE_SUPPORT

    //get_noe_type_field_axes_oor_ndarrays_fct_ = &detail::get_field_axes_oor_ndarrays<uintptr_t>;

    // Initialize the bin content array with objects using their default
    // constructor when the bin content array is an object array.
    if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
    {
        bp::object owner;
        bn::ndarray bc_arr = bc_->construct_ndarray(bc_->get_dtype(), 0, &owner);
        bn::iterators::flat_iterator< detail::bin_value_type_traits<bp::object> > bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);
        while(! bc_iter.is_end())
        {
            detail::bin_value_type_traits<bp::object>::value_ref_type bin = *bc_iter;

            bp::object sow_obj  = bc_class_();
            bp::object sows_obj = bc_class_();
            *bin.sow_obj_ptr_  = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sow_obj.ptr()));
            *bin.sows_obj_ptr_ = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sows_obj.ptr()));

            ++bc_iter;
        }
    }
}

ndhist &
ndhist::operator+=(ndhist const & rhs)
{
    iadd_fct_(*this, rhs);
    return *this;
}

ndhist
ndhist::
operator+(ndhist const & rhs) const
{
    if(! this->is_compatible(rhs))
    {
        std::stringstream ss;
        ss << "The right-hand-side ndhist object for the + operator must be "
           << "compatible with this ndhist object!";
        throw AssertionError(ss.str());
    }

    ndhist newhist = this->empty_like();
    newhist += *this;
    newhist += rhs;

    return newhist;
}

/*
ndhist
ndhist::
operator[](bp::object dim_slices) const
{
    // TODO: convert single object to list of length 1 first.
    if(! PyTuple_Check(dim_slices.ptr()))
    {
        dim_slices = bp::make_tuple(dim_slices);
    }

    // Determine the bin edges arrays of the new histogram.
    uintptr_t const dim = bp::len(dim_slices);
    if(dim > nd_)
    {
        std::stringstream ss;
        ss << "The number of given slices must not exceed "<< nd_ << "!";
        throw IndexError(ss.str());
    }

    if(dim < nd_)
    {
        // If all specified axes are intergers, the user has selected a certain
        // sub-histogram of this histogram which those indices fixed and full
        // axis range for the remaining axes.

        // But if one of 
    }

    std::cout << "Got "<< dim << " axes indices." << std::endl;
    bp::list newaxes_list;
    for(uintptr_t i=0; i<dim; ++i)
    {
        bp::object dim_slice_obj = dim_slices[i];
        if(bn::is_any_scalar(dim_slice_obj))
        {
            intptr_t const start = bp::extract<intptr_t>(dim_slice_obj);
            std::cout << "Got scalar '"<<start<<"' for axis '"<<i<<"'"<<std::endl;
            dim_slice_obj = bp::slice(start, start+1, 1);
        }

        if(detail::py::are_same_type_objects(dim_slice_obj, bp::slice()))
        {
            // Note: The dim_slice_obj is supposed to include the underflow (0) and
            // overflow bin (nbin) but the edges array does not have an explicit
            // underflow (-inf) and overflow (+inf) edge.
            uintptr_t const axis_size = axes_[i]->get_n_bins_fct(axes_[i]->data_);

            bp::slice axis_slice = bp::extract<bp::slice>(dim_slice_obj);
            detail::axis_index_iter axis_idx_iter(axis_size, detail::axis::UNDERFLOW_INDEX, detail::axis::OVERFLOW_INDEX);
            detail::axis_index_iter axis_idx_iter_end = axis_idx_iter.end();
            bp::slice::range<detail::axis_index_iter> axis_iter_range = axis_slice.get_indices(axis_idx_iter, axis_idx_iter_end);

            std::cout << "axis="<<i<<", sliced indices=";
            while(axis_iter_range.start != axis_iter_range.stop)
            {
                intptr_t idx = *axis_iter_range.start;
                if(idx == detail::axis::UNDERFLOW_INDEX) {
                    std::cout << "U,";
                }
                else if(idx == detail::axis::OVERFLOW_INDEX) {
                    std::cout << "O,";
                }
                else {
                    std::cout << idx << ",";
                }
                std::advance(axis_iter_range.start, axis_iter_range.step);
            }
            intptr_t idx = *axis_iter_range.start;
            if(idx == detail::axis::UNDERFLOW_INDEX) {
                std::cout << "U,";
            }
            else if(idx == detail::axis::OVERFLOW_INDEX) {
                std::cout << "O,";
            }
            else {
                std::cout << idx << ",";
            }

            std::cout << std::endl << std::flush;
        }
        else if(bn::is_ndarray(dim_slice_obj))
        {
            throw TypeError("ndarray indexing is not supported yet.");
        }
        else
        {
            std::stringstream ss;
            ss << "The dimension slices must be given either as integer or "
               << "slice objects!";
            throw IndexError(ss.str());
        }





        // Create the axis definition for the new sub-axis.
//         bn::ndarray edges_arr = axes_[i]->get_edges_ndarray_fct(axes_[i]->data_);
//         bn::ndarray subedges_arr = edges_arr[dim_slice_obj];
//         newaxes_list.append(get_axis_definition(i, subedges_arr));
    }
    bp::tuple newaxes(newaxes_list);
    return empty_like();
}
*/

bool
ndhist::
is_compatible(ndhist const & other) const
{
    if(nd_ != other.nd_) {
        return false;
    }
    for(uintptr_t i=0; i<nd_; ++i)
    {
        bn::ndarray const this_axis_edges_arr = this->axes_[i]->get_edges_ndarray();
        bn::ndarray const other_axis_edges_arr = other.axes_[i]->get_edges_ndarray();

        if(bn::all(bn::equal(this_axis_edges_arr, other_axis_edges_arr), 0) == false)
        {
            return false;
        }
    }

    return true;
}

ndhist
ndhist::
empty_like() const
{
    bp::list axis_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        axis_list.append(axes_[i]);
    }
    bp::tuple axes(axis_list);
    return ndhist(axes, bc_weight_dt_, bc_class_);
}

// ndhist
// ndhist::
// project(bp::object const & dims) const
// {
//     intptr_t const nd = get_nd();
//     bn::ndarray axes_arr = bn::from_object(dims, bn::dtype::get_builtin<intptr_t>(), 0, 1, bn::ndarray::ALIGNED);
//     std::set<intptr_t> axes;
//     bn::iterators::flat_iterator< bn::iterators::single_value<intptr_t> > axes_arr_iter(axes_arr);
//     while(! axes_arr_iter.is_end())
//     {
//         intptr_t axis = *axes_arr_iter;
//         if(axis < 0) {
//             axis += nd;
//         }
//         if(axis < 0)
//         {
//             std::stringstream ss;
//             ss << "The axis value \""<< *axes_arr_iter <<"\" specifies an "
//                << "axis < 0!";
//             throw IndexError(ss.str());
//         }
//         else if(axis >= nd)
//         {
//             std::stringstream ss;
//             ss << "The axis value \""<< axis <<"\" must be smaller than the "
//                << "dimensionality of the histogram, i.e. smaller than "
//                << nd <<"!";
//             throw IndexError(ss.str());
//         }
//         if(! axes.insert(axis).second)
//         {
//             std::stringstream ss;
//             ss << "The axis value \""<< axis <<"\" has been "
//                << "specified at least twice!";
//             throw ValueError(ss.str());
//         }
//         ++axes_arr_iter;
//     }
//     return project_fct_(*this, axes);
// }

bp::tuple
ndhist::
py_get_nbins() const
{
    std::vector<intptr_t> const & shape = bc_->get_shape_vector();
    std::vector<intptr_t> nbins(nd_);
    for(uintptr_t i=0; i<nd_; ++i)
    {
        nbins[i] = (axes_[i]->is_extendable() ? shape[i] : shape[i]-2);
    }
    bp::list shape_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        shape_list.append(nbins[i]);
    }
    bp::tuple shape_tuple(shape_list);
    return shape_tuple;
}

bn::ndarray
ndhist::
py_get_noe_ndarray()
{
    return bc_->construct_ndarray(bc_noe_dt_, 0);
}

bn::ndarray
ndhist::
py_get_sow_ndarray()
{
    return bc_->construct_ndarray(bc_weight_dt_, 1);
}

bn::ndarray
ndhist::
py_get_sows_ndarray()
{
    return bc_->construct_ndarray(bc_weight_dt_, 2);
}

bp::tuple
ndhist::
py_get_labels() const
{
    bp::list labels_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        labels_list.append(axes_[i]->get_label());
    }
    bp::tuple labels_tuple(labels_list);
    return labels_tuple;
}

// bp::tuple
// ndhist::
// py_get_underflow_entries() const
// {
//     std::vector<bn::ndarray> array_vec = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_UNDERFLOW, 0);
//
//     bp::list underflow_list;
//     for(size_t i=0; i<array_vec.size(); ++i)
//     {
//         underflow_list.append(array_vec[i]);
//     }
//     bp::tuple underflow(underflow_list);
//     return underflow;
// }

// bp::tuple
// ndhist::
// py_get_overflow_entries() const
// {
//     std::vector<bn::ndarray> array_vec = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_OVERFLOW, 0);
//
//     bp::list overflow_list;
//     for(size_t i=0; i<array_vec.size(); ++i)
//     {
//         overflow_list.append(array_vec[i]);
//     }
//     bp::tuple overflow(overflow_list);
//     return overflow;
// }

// bp::tuple
// ndhist::
// py_get_underflow() const
// {
//     std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_UNDERFLOW, 1);
//
//     bp::list underflow_list;
//     for(size_t i=0; i<array_vec.size(); ++i)
//     {
//         underflow_list.append(array_vec[i]);
//     }
//     bp::tuple underflow(underflow_list);
//     return underflow;
// }

// bp::tuple
// ndhist::
// py_get_overflow() const
// {
//     std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_OVERFLOW, 1);
//
//     bp::list overflow_list;
//     for(size_t i=0; i<array_vec.size(); ++i)
//     {
//         overflow_list.append(array_vec[i]);
//     }
//     bp::tuple overflow(overflow_list);
//     return overflow;
// }

// bp::tuple
// ndhist::
// py_get_underflow_squaredweights() const
// {
//     std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_UNDERFLOW, 2);
//
//     bp::list underflow_list;
//     for(size_t i=0; i<array_vec.size(); ++i)
//     {
//         underflow_list.append(array_vec[i]);
//     }
//     bp::tuple underflow(underflow_list);
//     return underflow;
// }

// bp::tuple
// ndhist::
// py_get_overflow_squaredweights() const
// {
//     std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_OVERFLOW, 2);
//
//     bp::list overflow_list;
//     for(size_t i=0; i<array_vec.size(); ++i)
//     {
//         overflow_list.append(array_vec[i]);
//     }
//     bp::tuple overflow(overflow_list);
//     return overflow;
// }

bn::ndarray
ndhist::
get_edges_ndarray(intptr_t axis) const
{
    // Count axis from the back if axis is negative.
    if(axis < 0) {
        axis += axes_.size();
    }

    if(axis < 0 || axis >= intptr_t(axes_.size()))
    {
        std::stringstream ss;
        ss << "The axis parameter must be in the interval "
           << "[0, " << axes_.size()-1 << "] or "
           << "[-1, -"<< axes_.size() <<"]!";
        throw IndexError(ss.str());
    }

    return axes_[axis]->get_edges_ndarray();
}

bp::object
ndhist::
py_get_binedges() const
{
    // Special case for 1d histograms, where we skip the extra tuple of
    // length 1.
    if(nd_ == 1)
    {
        return get_edges_ndarray(0);
    }

    bp::list edges_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        bn::ndarray edges = get_edges_ndarray(i);
        edges_list.append(edges);
    }
    bp::tuple edges_tuple(edges_list);
    return edges_tuple;
}

// void
// ndhist::
// fill(bp::object const & ndvalue_obj, bp::object weight_obj)
// {
//     // In case None is given as weight, we will use one.
//     if(weight_obj == bp::object())
//     {
//         weight_obj = bp::object(1);
//     }
//     fill_fct_(*this, ndvalue_obj, weight_obj);
// }

// TODO: Use the new multi_axis_iter for this.
static
void
initialize_extended_array_axis_range(
    bn::iterators::flat_iterator< bn::iterators::single_value<bp::object> > & iter
  , intptr_t axis
  , std::vector<intptr_t> const & shape
  , std::vector<intptr_t> const & strides
  , intptr_t n_iters
  , intptr_t axis_idx_range_min
  , intptr_t axis_idx_range_max
  , bp::object const obj_class
)
{
    int const nd = strides.size();

    intptr_t const last_axis = (nd - 1 == axis ? nd - 2 : nd - 1);
    //std::cout << "last_axis = "<< last_axis << std::endl<<std::flush;
    std::vector<intptr_t> indices(nd);
    for(intptr_t axis_idx=axis_idx_range_min; axis_idx < axis_idx_range_max; ++axis_idx)
    {
        //std::cout << "Start new axis idx"<<std::endl<<std::flush;
        memset(&indices.front(), 0, nd*sizeof(intptr_t));
        indices[axis] = axis_idx;
        // The iteration follows a matrix. The index pointer p indicates
        // index that needs to be incremented.
        // We need to start from the innermost dimension, unless it is the
        // iteration axis.
        intptr_t p = last_axis;
        for(intptr_t i=0; i<n_iters; ++i)
        {
            //std::cout << "indices = ";
            //for(intptr_t j=0; j<nd; ++j)
            //{
                //std::cout << indices[j] << ",";
            //}
            //std::cout << std::endl;

            intptr_t iteridx = 0;
            for(intptr_t j=nd-1; j>=0; --j)
            {
                iteridx += indices[j]*strides[j];
            }
            //std::cout << "iteridx = " << iteridx << std::endl<<std::flush;
            iter.jump_to_iter_index(iteridx);
            //std::cout << "jump done" << std::endl<<std::flush;

            *iter;
            uintptr_t * obj_ptr_ptr = iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            //std::cout << "Setting pointer data ..."<<std::flush;
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
            //std::cout << "done."<<std::endl<<std::flush;
            if(i == n_iters-1) break;
            // Move the index pointer to the next outer-axis if the index
            // of the current axis has reached its maximum. Then increase
            // the index and reset all indices to the right of this
            // increased index to zero. After this operation, the index
            // pointer points to the inner-most axis (excluding the
            // iteration axis).
            //std::cout << "p1 = "<<p<<std::endl<<std::flush;
            while(indices[p] == shape[p]-1)
            {
                --p;
                if(p == axis) --p;
            }
            //std::cout << "p2 = "<<p<<std::endl<<std::flush;
            indices[p]++;
            while(p < last_axis)
            {
                ++p;
                if(p == axis) ++p;
                indices[p] = 0;
            }

            //std::cout << "p3 = "<<p<<std::endl;
        }
    }
}

void
ndhist::
initialize_extended_array_axis(
    bp::object & arr_obj
  , bp::object const & obj_class
  , intptr_t axis
  , intptr_t f_n_extra_bins
  , intptr_t b_n_extra_bins
)
{
    if(f_n_extra_bins == 0 && b_n_extra_bins == 0) return;

    bn::ndarray & arr = *static_cast<bn::ndarray *>(&arr_obj);
    std::vector<intptr_t> const shape = arr.get_shape_vector();

    intptr_t f_axis_idx_range_min = 0;
    intptr_t f_axis_idx_range_max = 0;
    if(f_n_extra_bins > 0)
    {
        f_axis_idx_range_min = 0;
        f_axis_idx_range_max = f_n_extra_bins;
    }

    intptr_t b_axis_idx_range_min = 0;
    intptr_t b_axis_idx_range_max = 0;
    if(b_n_extra_bins > 0)
    {
        b_axis_idx_range_min = shape[axis] - b_n_extra_bins;
        b_axis_idx_range_max = shape[axis];
    }

    bn::iterators::flat_iterator< bn::iterators::single_value<bp::object> > bc_iter(arr, bn::detail::iter_operand::flags::WRITEONLY::value);
    intptr_t const nd = arr.get_nd();
    if(nd == 1)
    {
        // We can just use the flat iterator directly.

        // --- for front elements.
        for(intptr_t axis_idx = f_axis_idx_range_min; axis_idx < f_axis_idx_range_max; ++axis_idx)
        {
            bc_iter.jump_to_iter_index(axis_idx);
            *bc_iter;
            uintptr_t * obj_ptr_ptr = bc_iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }

        // --- for back elements.
        for(intptr_t axis_idx = b_axis_idx_range_min; axis_idx < b_axis_idx_range_max; ++axis_idx)
        {
            bc_iter.jump_to_iter_index(axis_idx);
            *bc_iter;
            uintptr_t * obj_ptr_ptr = bc_iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }
    }
    else
    {
        // Create a strides vector for the iteration.
        std::vector<intptr_t> strides(nd);
        strides[nd-1] = 1;
        for(intptr_t i=nd-2; i>=0; --i)
        {
            strides[i] = shape[i+1]*strides[i+1];
        }

        // Calculate the number of iterations (without the iteration axis).
        //std::cout << "shape = ";
        intptr_t n_iters = 1;
        for(intptr_t i=0; i<nd; ++i)
        {
            //std::cout << shape[i] << ",";
            if(i != axis) {
                n_iters *= shape[i];
            }
        }
        //std::cout << std::endl;
        //std::cout << "n_iters = " << n_iters << std::endl<<std::flush;

        // Initialize the front elements.
        initialize_extended_array_axis_range(bc_iter, axis, shape, strides, n_iters, f_axis_idx_range_min, f_axis_idx_range_max, obj_class);
        // Initialize the back elements.
        initialize_extended_array_axis_range(bc_iter, axis, shape, strides, n_iters, b_axis_idx_range_min, b_axis_idx_range_max, obj_class);
    }
}

void
ndhist::
extend_axes(
    std::vector<intptr_t> const & f_n_extra_bins_vec
  , std::vector<intptr_t> const & b_n_extra_bins_vec
)
{
    for(uintptr_t i=0; i<nd_; ++i)
    {
        boost::shared_ptr<Axis> & axis = this->axes_[i];
        axis->extend(f_n_extra_bins_vec[i], b_n_extra_bins_vec[i]);
    }
}

void
ndhist::
extend_bin_content_array(
    std::vector<intptr_t> const & f_n_extra_bins_vec
  , std::vector<intptr_t> const & b_n_extra_bins_vec
)
{
    // Extend the bin content array. This might cause a reallocation of memory.
    bc_->extend_axes(f_n_extra_bins_vec, b_n_extra_bins_vec, axes_extension_max_fcap_vec_, axes_extension_max_bcap_vec_);

    // We need to initialize the new bin content values, if the data type
    // is object.
    if(! bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
        return;

    bp::object owner;
    bn::ndarray bc_sow_arr  = bc_->construct_ndarray(bc_weight_dt_, 1, &owner);
    bn::ndarray bc_sows_arr = bc_->construct_ndarray(bc_weight_dt_, 2, &owner);
    for(uintptr_t axis=0; axis<nd_; ++axis)
    {
        initialize_extended_array_axis(bc_sow_arr,  bc_class_, axis, f_n_extra_bins_vec[axis], b_n_extra_bins_vec[axis]);
        initialize_extended_array_axis(bc_sows_arr, bc_class_, axis, f_n_extra_bins_vec[axis], b_n_extra_bins_vec[axis]);
    }
}

}//namespace ndhist
