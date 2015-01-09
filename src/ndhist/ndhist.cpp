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

#include <boost/type_traits/is_same.hpp>

#include <boost/python/list.hpp>
#include <boost/python/refcount.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/indexed_iterator.hpp>
#include <boost/numpy/iterators/multi_flat_iterator.hpp>
#include <boost/numpy/dstream.hpp>
#include <boost/numpy/utilities.hpp>

#include <ndhist/limits.hpp>
#include <ndhist/ndhist.hpp>
#include <ndhist/detail/axis.hpp>
#include <ndhist/detail/limits.hpp>
#include <ndhist/detail/generic_axis.hpp>
#include <ndhist/detail/constant_bin_width_axis.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

namespace detail {

template <typename AxisValueType>
struct axis_traits
{
    static bool
    has_constant_bin_width(bn::ndarray const & edges)
    {
        bn::flat_iterator<AxisValueType> edges_iter(edges);
        bn::flat_iterator<AxisValueType> const edges_iter_end(edges_iter.end());
        AxisValueType prev_value = *edges_iter;
        ++edges_iter;
        bool is_first_dist = true;
        AxisValueType first_dist;
        for(; edges_iter != edges_iter_end; ++edges_iter)
        {
            AxisValueType this_value = *edges_iter;
            AxisValueType this_dist = this_value - prev_value;
            if(is_first_dist)
            {
                prev_value = this_value;
                first_dist = this_dist;
                is_first_dist = false;
            }
            else
            {
                if(this_dist == first_dist)
                {
                    prev_value = this_value;
                }
                else
                {
                    return false;
                }
            }
        }

        return true;
    }

    static
    boost::shared_ptr<Axis>
    construct_axis(
        ndhist * self
      , bn::ndarray const & edges
      , std::string const & label
      , intptr_t autoscale_fcap=0
      , intptr_t autoscale_bcap=0
    )
    {
        // Check if the edges have a constant bin width,
        // thus the axis is linear.
        if(has_constant_bin_width(edges))
        {
            //std::cout << "+++++++++++++ Detected const bin width of "  << std::endl;
            return boost::shared_ptr<detail::ConstantBinWidthAxis<AxisValueType> >(new detail::ConstantBinWidthAxis<AxisValueType>(edges, label, autoscale_fcap, autoscale_bcap));
        }

        return boost::shared_ptr< detail::GenericAxis<AxisValueType> >(new detail::GenericAxis<AxisValueType>(self, edges, label));
    }

};

template <>
struct axis_traits<bp::object>
{
    static
    boost::shared_ptr<Axis>
    construct_axis(
        ndhist * self
      , bn::ndarray const & edges
      , std::string const & label
      , intptr_t
      , intptr_t
    )
    {
        // In case we have an object value typed axis, we use the
        // GenericAxis, because it requires only the < comparison
        // operator.
        return boost::shared_ptr< detail::GenericAxis<bp::object> >(new detail::GenericAxis<bp::object>(self, edges, label));
    }
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
    set_value_from_data(char * dst_addr, char * src_addr)
    {
        uintptr_t * dst_obj_ptr_ptr = reinterpret_cast<uintptr_t *>(dst_addr);
        uintptr_t * src_obj_ptr_ptr = reinterpret_cast<uintptr_t *>(src_addr);
        bp::object src_obj(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*src_obj_ptr_ptr)));
        *dst_obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(src_obj.ptr()));
    }
};

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
struct bin_value_type_traits
  : bn::iterators::value_type_traits
{
    typedef bin_value_type_traits<BCValueType>
            type_t;

    typedef bin_value<BCValueType>
            value_type;
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
    void
    dereference(
        bn::iterators::value_type_traits & vtt_base
      , value_ptr_type & value_ptr_ref
      , char * data_ptr
    )
    {
        type_t & vtt = *static_cast<type_t *>(&vtt_base);

        vtt.bin_value_.noe_  = reinterpret_cast<uintptr_t *>(data_ptr);
        vtt.bin_value_.sow_  = reinterpret_cast<BCValueType *>(data_ptr + vtt.fields_byte_offsets_[1]);
        vtt.bin_value_.sows_ = reinterpret_cast<BCValueType *>(data_ptr + vtt.fields_byte_offsets_[2]);

        value_ptr_ref = &vtt.bin_value_;
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
    void
    dereference(
        bn::iterators::value_type_traits & vtt_base
      , value_ptr_type & value_ptr_ref
      , char * data_ptr
    )
    {
        type_t & vtt = *static_cast<type_t *>(&vtt_base);
        vtt.bin_value_.noe_  = reinterpret_cast<uintptr_t*>(data_ptr);

        vtt.bin_value_.sow_obj_ptr_ = reinterpret_cast<uintptr_t*>(data_ptr + vtt.fields_byte_offsets_[1]);
        bp::object sow_obj(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*vtt.bin_value_.sow_obj_ptr_)));
        vtt.bin_value_.sow_obj_ = sow_obj;
        vtt.bin_value_.sow_ = &vtt.bin_value_.sow_obj_;

        vtt.bin_value_.sows_obj_ptr_ = reinterpret_cast<uintptr_t*>(data_ptr + vtt.fields_byte_offsets_[2]);
        bp::object sows_obj(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*vtt.bin_value_.sows_obj_ptr_)));
        vtt.bin_value_.sows_obj_ = sows_obj;
        vtt.bin_value_.sows_ = &vtt.bin_value_.sows_obj_;

        value_ptr_ref = &vtt.bin_value_;
    }
};

template <typename BCValueType>
static
void
flush_oor_cache(
    ndhist                & self
  , std::vector<intptr_t> & f_n_extra_bins_vec
  , uintptr_t               bc_data_offset
  , std::vector<intptr_t> & bc_data_strides
  , OORFillRecordStack<BCValueType> & oorfrstack
)
{
    typedef typename bc_value_traits<BCValueType>::ref_type
            bc_ref_type;

    intptr_t const nd = f_n_extra_bins_vec.size();

    // Fill in the cached values.
    char * bc_data_addr;
    intptr_t idx = oorfrstack.get_size();
    while(idx--)
    {
        typename OORFillRecordStack<BCValueType>::oor_fill_record_type const & rec = oorfrstack.get_record(idx);
        if(rec.is_oor)
        {
            boost::shared_ptr<detail::ndarray_storage> & oor_arr = self.oor_arr_vec_[rec.oor_arr_idx];

            std::vector<intptr_t> oor_strides = oor_arr->CalcDataStrides();
            char * oor_data_addr = oor_arr->data_ + oor_arr->CalcDataOffset(0);
            for(size_t i=0; i<rec.oor_arr_noor_relative_indices_size; ++i)
            {
                oor_data_addr += (rec.oor_arr_noor_relative_indices[i] + f_n_extra_bins_vec[rec.oor_arr_noor_axes_indices[i]]) * oor_strides[i];
            }
            for(size_t i=0; i<rec.oor_arr_oor_relative_indices_size; ++i)
            {
                oor_data_addr += rec.oor_arr_oor_relative_indices[i] * oor_strides[rec.oor_arr_noor_relative_indices_size + i];
            }

            detail::bc_value_traits<BCValueType>::increment_bin(oor_data_addr, rec.weight);
        }
        else
        {
            // Translate the relative indices into an absolute
            // data address for the extended bin content array.
            bc_data_addr = self.bc_->data_ + bc_data_offset;
            for(intptr_t axis=0; axis<nd; ++axis)
            {
                bc_data_addr += (f_n_extra_bins_vec[axis] + rec.relative_indices[axis]) * bc_data_strides[axis];
            }

            detail::bc_value_traits<BCValueType>::increment_bin(bc_data_addr, rec.weight);
        }
    }

    // Finally, clear the stack.
    oorfrstack.clear();
}

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

        // Add the not-oor bin contents of the two ndhist objects.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_value_type_traits<BCValueType>
                  , bin_value_type_traits<BCValueType>
                >
                multi_iter_t;

        bp::object data_owner;
        bn::ndarray self_bc_arr = self.bc_->ConstructNDArray(self.bc_->get_dtype(), 0, &data_owner);
        bn::ndarray other_bc_arr = other.bc_->ConstructNDArray(other.bc_->get_dtype(), 0, &data_owner);
        multi_iter_t bc_it(
            self_bc_arr
          , other_bc_arr
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value);
        while(! bc_it.is_end())
        {
            bc_it.dereference();
            typename multi_iter_t::value_type_0 & self_bin_value = *bc_it.value_ptr_0;
            typename multi_iter_t::value_type_1 & other_bin_value = *bc_it.value_ptr_1;
            *self_bin_value.noe_  += *other_bin_value.noe_;
            *self_bin_value.sow_  += *other_bin_value.sow_;
            *self_bin_value.sows_ += *other_bin_value.sows_;
            bc_it.increment();
        }

        // Add the oor bin contents of the two ndhist objects.
        for(size_t i=0; i<self.oor_arr_vec_.size(); ++i)
        {
            bn::ndarray self_oor_arr = self.oor_arr_vec_[i]->ConstructNDArray(self.oor_arr_vec_[i]->get_dtype(), 0, &data_owner);
            bn::ndarray other_oor_arr = other.oor_arr_vec_[i]->ConstructNDArray(other.oor_arr_vec_[i]->get_dtype(), 0, &data_owner);

            multi_iter_t oor_it(
                self_oor_arr
              , other_oor_arr
              , boost::numpy::detail::iter_operand::flags::READWRITE::value
              , boost::numpy::detail::iter_operand::flags::READONLY::value);

            while(! oor_it.is_end())
            {
                oor_it.dereference();
                typename multi_iter_t::value_type_0 & self_bin_value = *oor_it.value_ptr_0;
                typename multi_iter_t::value_type_1 & other_bin_value = *oor_it.value_ptr_1;
                *self_bin_value.noe_  += *other_bin_value.noe_;
                *self_bin_value.sow_  += *other_bin_value.sow_;
                *self_bin_value.sows_ += *other_bin_value.sows_;
                oor_it.increment();
            }
        }
    }
};


template <typename BCValueType>
struct imul_fct_traits
{
    static
    void apply(ndhist & self, bn::ndarray const & value_arr)
    {
        // Multiply the not-oor bin contents of the ndhist object with the
        // scalar value.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_value_type_traits<BCValueType>
                  , bn::iterators::single_value<BCValueType>
                >
                multi_iter_t;

        bp::object data_owner;
        bn::ndarray self_bc_arr = self.bc_->ConstructNDArray(self.bc_->get_dtype(), 0, &data_owner);
        multi_iter_t bc_it(
            self_bc_arr
          , const_cast<bn::ndarray &>(value_arr)
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value
        );
        while(! bc_it.is_end())
        {
            bc_it.dereference();
            typename multi_iter_t::value_type_0 & self_bin_value = *bc_it.value_ptr_0;
            typename multi_iter_t::value_type_1 & value          = *bc_it.value_ptr_1;
            *self_bin_value.sow_  *= value;
            *self_bin_value.sows_ *= value * value;
            ++bc_it;
        }

        // Multiply the oor bin contents of the ndhist object by the
        // scalar value.
        for(size_t i=0; i<self.oor_arr_vec_.size(); ++i)
        {
            bn::ndarray self_oor_arr = self.oor_arr_vec_[i]->ConstructNDArray(self.oor_arr_vec_[i]->get_dtype(), 0, &data_owner);

            multi_iter_t oor_it(
                self_oor_arr
              , const_cast<bn::ndarray &>(value_arr)
              , boost::numpy::detail::iter_operand::flags::READWRITE::value
              , boost::numpy::detail::iter_operand::flags::READONLY::value);

            while(! oor_it.is_end())
            {
                oor_it.dereference();
                typename multi_iter_t::value_type_0 & self_bin_value = *oor_it.value_ptr_0;
                typename multi_iter_t::value_type_1 & value          = *oor_it.value_ptr_1;
                *self_bin_value.sow_  *= value;
                *self_bin_value.sows_ *= value * value;
                ++oor_it;
            }
        }
    }
};


template <typename BCValueType>
struct idiv_fct_traits
{
    static
    void apply(ndhist & self, bn::ndarray const & value_arr)
    {
        // Divide the not-oor bin contents of the ndhist object with the
        // scalar value.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_value_type_traits<BCValueType>
                  , bn::iterators::single_value<BCValueType>
                >
                multi_iter_t;

        bp::object data_owner;
        bn::ndarray self_bc_arr = self.bc_->ConstructNDArray(self.bc_->get_dtype(), 0, &data_owner);
        multi_iter_t bc_it(
            self_bc_arr
          , const_cast<bn::ndarray &>(value_arr)
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value
        );
        while(! bc_it.is_end())
        {
            bc_it.dereference();
            typename multi_iter_t::value_type_0 & self_bin_value = *bc_it.value_ptr_0;
            typename multi_iter_t::value_type_1 & value          = *bc_it.value_ptr_1;
            *self_bin_value.sow_  /= value;
            *self_bin_value.sows_ /= value * value;
            ++bc_it;
        }

        // Divide the oor bins contents of the ndhist object by the
        // scalar value.
        for(size_t i=0; i<self.oor_arr_vec_.size(); ++i)
        {
            bn::ndarray self_oor_arr = self.oor_arr_vec_[i]->ConstructNDArray(self.oor_arr_vec_[i]->get_dtype(), 0, &data_owner);

            multi_iter_t oor_it(
                self_oor_arr
              , const_cast<bn::ndarray &>(value_arr)
              , boost::numpy::detail::iter_operand::flags::READWRITE::value
              , boost::numpy::detail::iter_operand::flags::READONLY::value);

            while(! oor_it.is_end())
            {
                oor_it.dereference();
                typename multi_iter_t::value_type_0 & self_bin_value = *oor_it.value_ptr_0;
                typename multi_iter_t::value_type_1 & value          = *oor_it.value_ptr_1;
                *self_bin_value.sow_  /= value;
                *self_bin_value.sows_ /= value * value;
                ++oor_it;
            }
        }
    }
};

template <typename BCValueType>
std::vector<bn::ndarray>
get_field_axes_oor_ndarrays(
    ndhist const & self
  , axis::out_of_range_t const oortype
  , size_t const field_idx
)
{
    uintptr_t const nd = self.get_nd();
    std::vector<bn::ndarray> array_vec;
    array_vec.reserve(nd);
    uintptr_t oor_arr_idx;
    uintptr_t noor_size;
    uintptr_t oor_size;
    std::vector<intptr_t> noor_indices(nd);
    std::vector<intptr_t> oor_indices(nd);
    std::vector<intptr_t> oor_strides(nd);
    std::vector<intptr_t> fields_byte_offsets = self.bc_->get_dtype().get_fields_byte_offsets();
    char * oor_data_addr;
    for(uintptr_t i=0; i<nd; ++i)
    {
        std::vector<intptr_t> const & histshape = self.bc_->get_shape_vector();
        std::vector<intptr_t> shape(nd);
        for(uintptr_t j=0; j<nd; ++j)
        {
            shape[j] = (j == i ? 1 : histshape[j] + 2);
        }

        bn::ndarray arr = bn::empty(shape, (field_idx == 0 ? self.bc_noe_dt_ : self.bc_weight_dt_));
        bn::indexed_iterator<BCValueType> iter(arr, bn::detail::iter_operand::flags::READWRITE::value);
        std::vector<intptr_t> indices(nd);
        while(! iter.is_end())
        {
            iter.get_multi_index_vector(indices);

            // Determine the oor array index and the data address of the bin.
            oor_arr_idx = 0;
            noor_size = 0;
            oor_size = 0;
            for(uintptr_t j=0; j<nd; ++j)
            {
                // Mark if the axis j is available.
                if(j != i)
                {
                    if(indices[j] == 0)
                    {
                        // It's the underflow bin of axis j.
                        oor_indices[oor_size] = 0;
                        ++oor_size;
                    }
                    else if(indices[j] == histshape[j]+1) // Notice the underflow element.
                    {
                        // It's the overflow bin of axis j.
                        oor_indices[oor_size] = 1;
                        ++oor_size;
                    }
                    else
                    {
                        // It's a normal bin of axis j.
                        oor_arr_idx |= (1<<j);
                        noor_indices[noor_size] = indices[j] - 1; // Notice the underflow element.
                        ++noor_size;
                    }
                }
                else
                {
                    // It's the out-of-range axis of interest.
                    oor_indices[oor_size] = (oortype == axis::OOR_UNDERFLOW ? 0 : 1);
                    ++oor_size;

                }
            }
            boost::shared_ptr<detail::ndarray_storage> const & oor_arr_storage = self.oor_arr_vec_[oor_arr_idx];
            oor_arr_storage->calc_data_strides(oor_strides);
            oor_data_addr = oor_arr_storage->data_ + oor_arr_storage->CalcDataOffset(fields_byte_offsets[field_idx]);
            for(size_t j=0; j<noor_size; ++j)
            {
                oor_data_addr += noor_indices[j]*oor_strides[j];
            }
            for(size_t j=0; j<oor_size; ++j)
            {
                oor_data_addr += oor_indices[j]*oor_strides[noor_size+j];
            }

            // Set the element of the new ndarray to the (sub) element of the
            // oor array.
            bc_value_traits<BCValueType>::set_value_from_data(iter.get_detail_iter().get_data(0), oor_data_addr);

            ++iter;
        }

        array_vec.push_back(arr);
    }

    return array_vec;
}


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
            bn::indexed_iterator<BCValueType> bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);

            // Do the iteration.
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

            // Trigger the recreation of temporary properties.
            self.recreate_oorpadded_bc_ = true;
        }
    }; // struct fill_traits
}; // struct generic_nd_traits

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
  , recreate_oorpadded_bc_(true)
{
    std::vector<intptr_t> shape(nd_);
    axes_extension_max_fcap_vec_.resize(nd_);
    axes_extension_max_bcap_vec_.resize(nd_);

    // Construct the axes of the histogram.
    for(size_t i=0; i<nd_; ++i)
    {
        // Each axes element can be a single ndarray or a tuple of the form
        // (edges_ndarry[, axis_name[, axis_title[, front_capacity, back_capacity]]])
        bp::object axis = axes[i];
        bp::object edges_arr_obj;
        bp::object axis_name_obj;
        bp::object axis_label_obj;
        bp::object fcap_obj;
        bp::object bcap_obj;
        if(PyTuple_Check(axis.ptr()))
        {
            size_t const tuple_len = bp::len(axis);
            if(tuple_len == 0)
            {
                std::stringstream ss;
                ss << "The "<<i<<"th axis tuple is empty!";
                throw ValueError(ss.str());
            }
            else if(tuple_len == 1)
            {
                edges_arr_obj = axis[0];
                std::stringstream axis_name;
                axis_name << "a" << i;
                axis_name_obj = bp::str(axis_name.str());
                axis_label_obj = bp::str("");
                fcap_obj = bp::object(0);
                bcap_obj = bp::object(0);
            }
            else if(tuple_len == 2)
            {
                edges_arr_obj = axis[0];
                axis_name_obj = bp::str(axis[1]);
                axis_label_obj = bp::str("");
                fcap_obj = bp::object(0);
                bcap_obj = bp::object(0);
            }
            else if(tuple_len == 3)
            {
                edges_arr_obj  = axis[0];
                axis_name_obj  = axis[1];
                axis_label_obj = axis[2];
                fcap_obj = bp::object(0);
                bcap_obj = bp::object(0);
            }
            else if(tuple_len == 5)
            {
                edges_arr_obj  = axis[0];
                axis_name_obj  = axis[1];
                axis_label_obj = axis[2];
                fcap_obj       = axis[3];
                bcap_obj       = axis[4];
            }
            else
            {
                std::stringstream ss;
                ss << "The "<<i<<"th axis tuple must have a length of "
                   << "either 1, 2, 3, or 5!";
                throw ValueError(ss.str());
            }
        }
        else
        {
            // Only the edges array is given.
            edges_arr_obj = axis;
            std::stringstream axis_name;
            axis_name << "a" << i;
            axis_name_obj = bp::str(axis_name.str());
            axis_label_obj = bp::str("");
            fcap_obj = bp::object(0);
            bcap_obj = bp::object(0);
        }

        bn::ndarray edges_arr = bn::from_object(edges_arr_obj, 0, 1, bn::ndarray::ALIGNED);
        if(edges_arr.get_nd() != 1)
        {
            std::stringstream ss;
            ss << "The dimension of the edges array for the " << i+1 << "th "
               << "dimension of this histogram must be 1!";
            throw ValueError(ss.str());
        }
        const intptr_t n_bin_dim = edges_arr.get_size();

        intptr_t const axis_extension_max_fcap = bp::extract<intptr_t>(fcap_obj);
        intptr_t const axis_extension_max_bcap = bp::extract<intptr_t>(bcap_obj);
        std::string const axis_label = bp::extract<std::string>(axis_label_obj);

        // Check the type of the edge values for the current axis.
        bool axis_dtype_supported = false;
        bn::dtype axis_dtype = edges_arr.get_dtype();
        #define NDHIST_AXIS_DATA_TYPE_SUPPORT(AXISDTYPE)                            \
            if(bn::dtype::equivalent(axis_dtype, bn::dtype::get_builtin<AXISDTYPE>()))\
            {                                                                       \
                if(axis_dtype_supported) {                                          \
                    std::stringstream ss;                                           \
                    ss << "The bin content data type is supported by more than one "\
                       << "possible C++ data type! This is an internal error!";     \
                    throw TypeError(ss.str());                                      \
                }                                                                   \
                axes_.push_back(detail::axis_traits<AXISDTYPE>::construct_axis(this, edges_arr, axis_label, axis_extension_max_fcap, axis_extension_max_bcap));\
                axis_dtype_supported = true;                                        \
            }
        NDHIST_AXIS_DATA_TYPE_SUPPORT(int8_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(uint8_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(int16_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(uint16_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(int32_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(uint32_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(int64_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(uint64_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(float)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(double)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(bp::object)
        if(!axis_dtype_supported)
        {
            std::stringstream ss;
            ss << "The data type of the edges of axis "<< i << " is not "
               << "supported.";
            throw TypeError(ss.str());
        }
        #undef NDHIST_AXIS_DATA_TYPE_SUPPORT

        // Add the axis field to the ndvalues dtype object.
        std::string field_name = bp::extract<std::string>(axis_name_obj);
        ndvalues_dt_.add_field(field_name, axes_[i]->get_dtype());

        // Add the bin content shape information for this axis.
        shape[i] = n_bin_dim - 1;

        // Set the extra front and back capacity for this axis if the axis has
        // an autoscale.
        if(axes_[i]->is_extendable())
        {
            axes_extension_max_fcap_vec_[i] = axis_extension_max_fcap;
            axes_extension_max_bcap_vec_[i] = axis_extension_max_bcap;
        }
        else
        {
            axes_extension_max_fcap_vec_[i] = 0;
            axes_extension_max_bcap_vec_[i] = 0;
        }
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
        #define NDHIST_BC_DATA_TYPE_SUPPORT(BCDTYPE)                           \
            if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<BCDTYPE>()))\
            {                                                                       \
                if(bc_dtype_supported) {                                            \
                    std::stringstream ss;                                           \
                    ss << "The bin content data type is supported by more than one "\
                    << "possible C++ data type! This is an internal error!";        \
                    throw TypeError(ss.str());                                      \
                }                                                                   \
                oor_fill_record_stack_ = boost::shared_ptr< detail::OORFillRecordStack<BCDTYPE> >(new detail::OORFillRecordStack<BCDTYPE>(nd_, oor_stack_size));\
                fill_fct_ = &detail::generic_nd_traits::fill_traits<BCDTYPE>::fill; \
                bc_dtype_supported = true;                                          \
            }
        NDHIST_BC_DATA_TYPE_SUPPORT(bool)
        NDHIST_BC_DATA_TYPE_SUPPORT(int8_t)
        NDHIST_BC_DATA_TYPE_SUPPORT(uint8_t)
        NDHIST_BC_DATA_TYPE_SUPPORT(int16_t)
        NDHIST_BC_DATA_TYPE_SUPPORT(uint16_t)
        NDHIST_BC_DATA_TYPE_SUPPORT(int32_t)
        NDHIST_BC_DATA_TYPE_SUPPORT(uint32_t)
        NDHIST_BC_DATA_TYPE_SUPPORT(int64_t)
        NDHIST_BC_DATA_TYPE_SUPPORT(uint64_t)
        NDHIST_BC_DATA_TYPE_SUPPORT(float)
        NDHIST_BC_DATA_TYPE_SUPPORT(double)
        NDHIST_BC_DATA_TYPE_SUPPORT(bp::object)
        #undef NDHIST_BC_DATA_TYPE_SUPPORT
    }
    if(!bc_dtype_supported)
    {
        std::stringstream ss;
        ss << "The data type of the bin content array is not supported.";
        throw TypeError(ss.str());
    }

    // Setup the function pointers, which depend on the bin content weight type.
    #define NDHIST_BC_DATA_TYPE_SUPPORT(BCDTYPE) \
        if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<BCDTYPE>()))\
        {                                                                       \
            iadd_fct_ = &detail::iadd_fct_traits<BCDTYPE>::apply;               \
            idiv_fct_ = &detail::idiv_fct_traits<BCDTYPE>::apply;               \
            imul_fct_ = &detail::imul_fct_traits<BCDTYPE>::apply;               \
            get_weight_type_field_axes_oor_ndarrays_fct_ = &detail::get_field_axes_oor_ndarrays<BCDTYPE>;\
        }
    NDHIST_BC_DATA_TYPE_SUPPORT(bool)
    NDHIST_BC_DATA_TYPE_SUPPORT(int8_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(uint8_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(int16_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(uint16_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(int32_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(uint32_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(int64_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(uint64_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(float)
    NDHIST_BC_DATA_TYPE_SUPPORT(double)
    NDHIST_BC_DATA_TYPE_SUPPORT(bp::object)
    #undef NDHIST_BC_DATA_TYPE_SUPPORT

    get_noe_type_field_axes_oor_ndarrays_fct_ = &detail::get_field_axes_oor_ndarrays<uintptr_t>;

    // Initialize the bin content array with objects using their default
    // constructor when the bin content array is an object array.
    if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
    {
        bp::object self(bp::ptr(this));
        bn::ndarray bc_sow_arr  = bc_->ConstructNDArray(bc_weight_dt_, 1, &self);
        bn::ndarray bc_sows_arr = bc_->ConstructNDArray(bc_weight_dt_, 2, &self);
        bn::flat_iterator<bp::object> bc_sow_iter(bc_sow_arr);
        bn::flat_iterator<bp::object> bc_sow_iter_end(bc_sow_iter.end());
        bn::flat_iterator<bp::object> bc_sows_iter(bc_sows_arr);
        for(; bc_sow_iter != bc_sow_iter_end; ++bc_sow_iter, ++bc_sows_iter)
        {
            uintptr_t * sow_obj_ptr_ptr = bc_sow_iter.get_object_ptr_ptr();
            uintptr_t * sows_obj_ptr_ptr = bc_sows_iter.get_object_ptr_ptr();
            bp::object sow_obj  = bc_class_();
            bp::object sows_obj = bc_class_();
            *sow_obj_ptr_ptr  = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sow_obj.ptr()));
            *sows_obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sows_obj.ptr()));
        }

        bc_one_ = bc_class(1);
    }
    else
    {
        bp::object one(1);
        bc_one_ = bn::from_object(one, bc_weight_dt_, 0, 1, bn::ndarray::ALIGNED).scalarize();
    }

    // Create the out-of-range (oor) arrays.
    create_oor_arrays(nd_, bc_dt, bc_weight_dt_, bc_class_);
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

bool
ndhist::
is_compatible(ndhist const & other) const
{
    if(nd_ != other.nd_) {
        return false;
    }
    for(uintptr_t i=0; i<nd_; ++i)
    {
        bn::ndarray const this_axis_edges_arr = this->axes_[i]->get_edges_ndarray_fct(this->axes_[i]->data_);
        bn::ndarray const other_axis_edges_arr = other.axes_[i]->get_edges_ndarray_fct(other.axes_[i]->data_);

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
    bp::tuple axes_name_tuple = ndvalues_dt_.get_field_names();
    bp::list axis_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        bp::tuple axis = bp::make_tuple(
            axes_[i]->get_edges_ndarray_fct(axes_[i]->data_)
          , axes_name_tuple[i]
          , axes_[i]->label_
          , axes_[i]->extension_max_fcap_
          , axes_[i]->extension_max_bcap_
        );
        axis_list.append(axis);
    }
    bp::tuple axes(axis_list);
    return ndhist(axes, bc_weight_dt_, bc_class_);
}

void
ndhist::
create_oor_arrays(
    uintptr_t nd
  , bn::dtype const & bc_dt
  , bn::dtype const & bc_weight_dt
  , bp::object const & bc_class
)
{
    bp::object self(bp::ptr(this));
    uintptr_t const n_arrays = std::pow(2, nd) - 1;
    oor_arr_vec_.reserve(n_arrays);
    for(uintptr_t idx=0; idx<n_arrays; ++idx)
    {
        //std::cout << "idx = " << idx << std::endl<<std::flush;
        std::bitset<NDHIST_LIMIT_MAX_ND> bset(idx);
        // Determine the shape of the nd-dim. array.
        std::vector<intptr_t> shape;
        std::vector<intptr_t> axes_extension_max_fcap_vec;
        std::vector<intptr_t> axes_extension_max_bcap_vec;
        shape.reserve(nd);
        axes_extension_max_fcap_vec.reserve(nd);
        axes_extension_max_bcap_vec.reserve(nd);
        // First set the shapes of the not-oor axes.
        for(uintptr_t i=0; i<nd; ++i)
        {
            if(bset.test(i))
            {
                //std::cout << "axis = " << i <<" bit is set."<< std::endl<<std::flush;
                boost::shared_ptr<detail::Axis> const & axis = axes_[i];
                shape.push_back(axis->get_n_bins_fct(axis->data_));
                axes_extension_max_fcap_vec.push_back(axis->extension_max_fcap_);
                axes_extension_max_bcap_vec.push_back(axis->extension_max_bcap_);
            }
        }
        // Now add the shape elements for the oor axes.
        uintptr_t i = nd - shape.size();
        while(i--)
        {
            shape.push_back(2);
            axes_extension_max_fcap_vec.push_back(0);
            axes_extension_max_bcap_vec.push_back(0);
            //std::cout << "Add 2 to shape." << std::endl<<std::flush;
        }
        // Now create the array.
        //std::cout << "Create arr " << std::endl<<std::flush;
        boost::shared_ptr<detail::ndarray_storage> arr_storage(new detail::ndarray_storage(shape, axes_extension_max_fcap_vec, axes_extension_max_bcap_vec, bc_dt));
        oor_arr_vec_.push_back(arr_storage);

        // In case the dtype is object, we need to initialize the array's values
        // with bc_class() objects.
        if(bn::dtype::equivalent(bc_weight_dt, bn::dtype::get_builtin<bp::object>()))
        {
            bn::flat_iterator<bp::object> iter_sow(arr_storage->ConstructNDArray(bc_weight_dt, 1, &self));
            bn::flat_iterator<bp::object> iter_sows(arr_storage->ConstructNDArray(bc_weight_dt, 2, &self));
            bn::flat_iterator<bp::object> iter_end(iter_sow.end());
            for(; iter_sow != iter_end; ++iter_sow, ++iter_sows)
            {
                uintptr_t * obj_ptr_ptr = iter_sow.get_object_ptr_ptr();
                bp::object obj_sow = bc_class();
                *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj_sow.ptr()));
                obj_ptr_ptr = iter_sows.get_object_ptr_ptr();
                bp::object obj_sows = bc_class();
                *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj_sows.ptr()));
            }
        }
    }
}

void
ndhist::
recreate_oorpadded_bc()
{
    bn::dtype const & bc_dt = bc_->get_dtype();
    std::vector<intptr_t> const & bc_shape = bc_->get_shape_vector();
    std::vector<intptr_t> shape(nd_);
    std::vector<intptr_t> fcap(nd_, 0);
    std::vector<intptr_t> bcap(nd_, 0);
    for(uintptr_t i=0; i<nd_; ++i)
    {
        shape[i] = bc_shape[i] + 2;
    }
    oorpadded_bc_ = boost::shared_ptr<detail::ndarray_storage>(new detail::ndarray_storage(shape, fcap, bcap, bc_dt));

    // Copy the bin content data over to the new storage.
    bp::object data_owner;
    bn::ndarray src_arr = bc_->ConstructNDArray(bc_dt, 0, &data_owner);
    std::vector<intptr_t> shape_offset_vec(nd_, 1);
    oorpadded_bc_->copy_from(src_arr, shape_offset_vec);

    // TODO: Copy also the out-of-range bins to their correct location.
}

bp::tuple
ndhist::
py_get_nbins() const
{
    std::vector<intptr_t> const & shape = bc_->get_shape_vector();
    bp::list shape_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        shape_list.append(shape[i]);
    }
    bp::tuple shape_tuple(shape_list);
    return shape_tuple;
}

bn::ndarray
ndhist::
py_get_noe_ndarray()
{
    return bc_->ConstructNDArray(bc_noe_dt_, 0);
}

bn::ndarray
ndhist::
py_get_sow_ndarray()
{
    return bc_->ConstructNDArray(bc_weight_dt_, 1);
}

bn::ndarray
ndhist::
py_get_oorpadded_sow_ndarray()
{
    // Check if we need to (re-)create the storage for this array.
    if(recreate_oorpadded_bc_)
    {
        recreate_oorpadded_bc();
        recreate_oorpadded_bc_ = false;
    }

    return oorpadded_bc_->ConstructNDArray(bc_weight_dt_, 1);
}

bn::ndarray
ndhist::
py_get_sows_ndarray()
{
    return bc_->ConstructNDArray(bc_weight_dt_, 2);
}

bn::ndarray
ndhist::
py_get_oorpadded_sows_ndarray()
{
    // Check if we need to (re-)create the storage for this array.
    if(recreate_oorpadded_bc_)
    {
        recreate_oorpadded_bc();
        recreate_oorpadded_bc_ = false;
    }

    return oorpadded_bc_->ConstructNDArray(bc_weight_dt_, 2);
}

bp::tuple
ndhist::
py_get_labels() const
{
    bp::list labels_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        labels_list.append(axes_[i]->label_);
    }
    bp::tuple labels_tuple(labels_list);
    return labels_tuple;
}

bp::tuple
ndhist::
py_get_underflow_entries() const
{
    std::vector<bn::ndarray> array_vec = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_UNDERFLOW, 0);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        underflow_list.append(array_vec[i]);
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_overflow_entries() const
{
    std::vector<bn::ndarray> array_vec = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_OVERFLOW, 0);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        overflow_list.append(array_vec[i]);
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}

bp::tuple
ndhist::
py_get_underflow() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_UNDERFLOW, 1);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        underflow_list.append(array_vec[i]);
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_overflow() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_OVERFLOW, 1);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        overflow_list.append(array_vec[i]);
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}

bp::tuple
ndhist::
py_get_underflow_squaredweights() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_UNDERFLOW, 2);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        underflow_list.append(array_vec[i]);
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_overflow_squaredweights() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, detail::axis::OOR_OVERFLOW, 2);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        overflow_list.append(array_vec[i]);
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}




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

    return axes_[axis]->get_edges_ndarray_fct(axes_[axis]->data_);
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

void
ndhist::
fill(bp::object const & ndvalue_obj, bp::object weight_obj)
{
    // In case None is given as weight, we will use one.
    if(weight_obj == bp::object())
    {
        weight_obj = this->get_one();
    }
    fill_fct_(*this, ndvalue_obj, weight_obj);
}

static
void
initialize_extended_array_axis_range(
    bn::flat_iterator<bp::object> & iter
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
            for(intptr_t j=0; j<nd; ++j)
            {
                //std::cout << indices[j] << ",";
            }
            //std::cout << std::endl;

            intptr_t iteridx = 0;
            for(intptr_t j=nd-1; j>=0; --j)
            {
                iteridx += indices[j]*strides[j];
            }
            //std::cout << "iteridx = " << iteridx << std::endl<<std::flush;
            iter.jump_to_iter_index(iteridx);
            //std::cout << "jump done" << std::endl<<std::flush;

            uintptr_t * obj_ptr_ptr = iter.get_object_ptr_ptr();
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

    bn::flat_iterator<bp::object> bc_iter(arr);
    intptr_t const nd = arr.get_nd();
    if(nd == 1)
    {
        // We can just use the flat iterator directly.

        // --- for front elements.
        for(intptr_t axis_idx = f_axis_idx_range_min; axis_idx < f_axis_idx_range_max; ++axis_idx)
        {
            bc_iter.jump_to_iter_index(axis_idx);
            uintptr_t * obj_ptr_ptr = bc_iter.get_object_ptr_ptr();
            bp::object obj = obj_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }

        // --- for back elements.
        for(intptr_t axis_idx = b_axis_idx_range_min; axis_idx < b_axis_idx_range_max; ++axis_idx)
        {
            bc_iter.jump_to_iter_index(axis_idx);
            uintptr_t * obj_ptr_ptr = bc_iter.get_object_ptr_ptr();
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
    int const nd = this->get_nd();
    for(int i=0; i<nd; ++i)
    {
        boost::shared_ptr<detail::Axis> & axis = this->axes_[i];
        axis->extend_fct(axis->data_, f_n_extra_bins_vec[i], b_n_extra_bins_vec[i]);
    }
}

void
ndhist::
extend_bin_content_array(
    std::vector<intptr_t> const & f_n_extra_bins_vec
  , std::vector<intptr_t> const & b_n_extra_bins_vec
)
{
    //std::cout << "extend_bin_content_array" << std::endl;
    bp::object self(bp::ptr(this));

    // Extend the bin content array. This might cause a reallocation of memory.
    bc_->extend_axes(f_n_extra_bins_vec, b_n_extra_bins_vec, axes_extension_max_fcap_vec_, axes_extension_max_bcap_vec_, &self);

    // We need to initialize the new bin content values, if the data type
    // is object.
    if(! bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
        return;

    bn::ndarray bc_sow_arr  = bc_->ConstructNDArray(bc_weight_dt_, 1, &self);
    bn::ndarray bc_sows_arr = bc_->ConstructNDArray(bc_weight_dt_, 2, &self);
    int const nd = this->get_nd();
    for(int axis=0; axis<nd; ++axis)
    {
        initialize_extended_array_axis(bc_sow_arr,  bc_class_, axis, f_n_extra_bins_vec[axis], b_n_extra_bins_vec[axis]);
        initialize_extended_array_axis(bc_sows_arr, bc_class_, axis, f_n_extra_bins_vec[axis], b_n_extra_bins_vec[axis]);
    }
}

}//namespace ndhist
