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
#include <cstddef>
#include <cstdlib>

#include <iostream>
#include <sstream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <ndhist/error.hpp>
#include <ndhist/detail/ndarray_storage.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace detail {

intptr_t
ndarray_storage::
calc_first_shape_element_data_offset(
    bn::dtype const & dt
  , std::vector<intptr_t> const & shape
  , std::vector<intptr_t> const & front_capacity
  , std::vector<intptr_t> const & back_capacity
  , intptr_t const sub_item_byte_offset
)
{
    intptr_t const nd = shape.size();
    if(nd == 0)
    {
        return sub_item_byte_offset;
    }
    intptr_t offset = front_capacity[nd-1];
    intptr_t dim_offsets = 1;
    for(intptr_t i=nd-2; i>=0; --i)
    {
        dim_offsets *= (front_capacity[i+1] + shape[i+1] + back_capacity[i+1]);
        offset += front_capacity[i]*dim_offsets;
    }
    offset *= dt.get_itemsize();

    return offset + sub_item_byte_offset;
}

void
ndarray_storage::
calc_data_strides(
    std::vector<intptr_t> & strides
  , bn::dtype const & dt
  , std::vector<intptr_t> const & shape
  , std::vector<intptr_t> const & front_capacity
  , std::vector<intptr_t> const & back_capacity
)
{
    size_t const nd = shape.size();
    if(nd == 0)
    {
        return;
    }
    strides[nd-1] = dt.get_itemsize();
    for(intptr_t i=nd-2; i>=0; --i)
    {
        strides[i] = (front_capacity[i+1] + shape[i+1] + back_capacity[i+1]) * strides[i+1];
    }
}

// This is the static function.
bn::ndarray
ndarray_storage::
construct_ndarray(
    ndarray_storage const & storage
  , bn::dtype const & dt
  , std::vector<intptr_t> const & shape
  , std::vector<intptr_t> const & front_capacity
  , std::vector<intptr_t> const & back_capacity
  , intptr_t const sub_item_byte_offset
  , bp::object const * data_owner
  , bool const set_owndata_flag
)
{
    intptr_t const data_offset = storage.bytearray_data_offset_ + calc_first_shape_element_data_offset(storage.get_dtype(), shape, front_capacity, back_capacity, sub_item_byte_offset);

    std::vector<intptr_t> strides(shape.size());
    calc_data_strides(strides, storage.get_dtype(), shape, front_capacity, back_capacity);

    return bn::from_data(storage.get_data() + data_offset, dt, shape, strides, data_owner, set_owndata_flag);
}

// This is the method.
bn::ndarray
ndarray_storage::
construct_ndarray(
    bn::dtype const & dt
  , size_t const field_idx
  , bp::object const * data_owner
  , bool set_owndata_flag
) const
{
    intptr_t const sub_item_byte_offset = (field_idx == 0 ? 0 : dt_.get_fields_byte_offsets()[field_idx]);
    intptr_t const data_offset = bytearray_data_offset_ + calc_first_shape_element_data_offset(dt_, shape_, front_capacity_, back_capacity_, sub_item_byte_offset);

    return bn::from_data(get_data() + data_offset, dt, shape_, data_strides_, data_owner, set_owndata_flag);
}

void
ndarray_storage::
extend_axes(
    std::vector<intptr_t> const & f_n_elements_vec
  , std::vector<intptr_t> const & b_n_elements_vec
  , std::vector<intptr_t> const & max_fcap_vec
  , std::vector<intptr_t> const & max_bcap_vec
)
{
    int const nd = this->get_nd();
    if(f_n_elements_vec.size() != size_t(nd) ||
       b_n_elements_vec.size() != size_t(nd)
      )
    {
        std::stringstream ss;
        ss << "The vector holding the number of new elements in front is "
           << f_n_elements_vec.size() <<" and the one for the back is "
           << b_n_elements_vec.size() <<", but both must be " << nd << "!";
        throw AssertionError(ss.str());
    }

    // First check if a memory reallocation is actually required.
    bool reallocate = false;
    for(int axis=0; axis<=nd; ++axis)
    {
        intptr_t const f_n_elements = f_n_elements_vec[axis];
        intptr_t const b_n_elements = b_n_elements_vec[axis];
        if(f_n_elements > 0)
        {
            if(front_capacity_[axis] - f_n_elements < 0)
            {
                reallocate = true;
            }
            shape_[axis] += f_n_elements;
            front_capacity_[axis] -= f_n_elements;
        }
        if(b_n_elements > 0)
        {
            if(back_capacity_[axis] - b_n_elements < 0)
            {
                reallocate = true;
            }
            shape_[axis] += b_n_elements;
            back_capacity_[axis] -= b_n_elements;
        }
    }

    if(reallocate)
    {
        std::vector<intptr_t> old_shape = shape_;
        std::vector<intptr_t> old_fcap = front_capacity_;
        std::vector<intptr_t> old_bcap = back_capacity_;
        for(int i=0; i<nd; ++i)
        {
            intptr_t const f_n_elements = f_n_elements_vec[i];
            intptr_t const b_n_elements = b_n_elements_vec[i];
            if(f_n_elements > 0)
            {
                old_shape[i] -= f_n_elements;
                old_fcap[i] += f_n_elements;
                front_capacity_[i] = max_fcap_vec[i];
            }
            if(b_n_elements > 0)
            {
                old_shape[i] -= b_n_elements;
                old_bcap[i] += b_n_elements;
                back_capacity_[i] = max_bcap_vec[i];
            }
        }

        intptr_t const itemsize = dt_.get_itemsize();

        // At this point shape_, front_capacity_ and back_capacity_ have the
        // right numbers for the new array.
        // Create a new bytearray.
        boost::shared_ptr<bytearray> new_bytearray = create_bytearray(shape_, front_capacity_, back_capacity_, itemsize);

        // Copy the data from the old memory to the new one. We do this by
        // creating two ndarray objects having the same layout. The first is the
        // old array layout on the new memory and the second is the old array on
        // the old memory. Then we just use the copy_into function to copy the
        // data.
        // We assume that the numpy people implemented the PyArray_CopyInto
        // C-API function efficient enough ;)

        // Calculate data offset of the old array inside the new memory.
        intptr_t new_offset = front_capacity_[nd-1] + f_n_elements_vec[nd-1];
        intptr_t new_dim_offsets = 1;
        for(int i=nd-2; i>=0; --i)
        {
            new_dim_offsets *= (front_capacity_[i+1]+f_n_elements_vec[i+1] + old_shape[i+1] + back_capacity_[i+1]+b_n_elements_vec[i+1]);
            new_offset += (front_capacity_[i]+f_n_elements_vec[i])*new_dim_offsets;
        }
        new_offset *= itemsize;

        // Calculate the strides of the old array inside the new memory.
        std::vector<intptr_t> new_strides(nd, itemsize);
        for(int i=nd-2; i>=0; --i)
        {
            new_strides[i] = (front_capacity_[i+1]+f_n_elements_vec[i+1] + old_shape[i+1] + back_capacity_[i+1]+b_n_elements_vec[i+1]) * new_strides[i+1];
        }

        // Create the old array from the new memory.
        bn::ndarray new_data_arr = bn::from_data(new_bytearray->get() + new_offset, dt_, old_shape, new_strides, /*owner=*/NULL, /*set_owndata_flag=*/false);

        // Calculate the data offset of the old array inside the old memory.
        intptr_t old_offset = old_fcap[nd-1];
        intptr_t old_dim_offsets = 1;
        for(int i=nd-2; i>=0; --i)
        {
            old_dim_offsets *= (old_fcap[i+1] + old_shape[i+1] + old_bcap[i+1]);
            old_offset += (old_fcap[i])*old_dim_offsets;
        }
        old_offset *= itemsize;

        // Calculate the data strides of the old array inside the old memory.
        std::vector<intptr_t> old_strides(nd, itemsize);
        for(int i=nd-2; i>=0; --i)
        {
            old_strides[i] = (old_fcap[i+1] + old_shape[i+1] + old_bcap[i+1]) * old_strides[i+1];
        }

        // Create the old array from the old memory.
        bn::ndarray old_data_arr = bn::from_data(bytearray_->get() + bytearray_data_offset_ + old_offset, dt_, old_shape, old_strides, /*owner=*/NULL, /*set_owndata_flag=*/false);

        // Now we just copy the old data to the new data.
        if(! bn::copy_into(new_data_arr, old_data_arr))
        {
            std::stringstream ss;
            ss << "Unable to copy the old data from the old memory into the "
               << "new memory!";
            throw MemoryError(ss.str());
        }

        // With the new allocated memory we don't have a vew data offset.
        bytearray_data_offset_ = 0;

        // Assign the new bytearray to this ndarray_storage. The old bytearray
        // will be destroyed automatically, if its boost::shared_ptr reference
        // count reaches zero.
        bytearray_ = new_bytearray;
    }

    // Recalculate the data strides.
    calc_data_strides(data_strides_, dt_, shape_, front_capacity_, back_capacity_);
}

void
ndarray_storage::
copy_from(
    bn::ndarray const & src_arr
  , std::vector<intptr_t> const & shape_offset_vec
)
{
    int const nd = get_nd();
    intptr_t const itemsize = dt_.get_itemsize();

    std::vector<intptr_t> const & dst_shape = get_shape_vector();
    std::vector<intptr_t> const src_shape = src_arr.get_shape_vector();

    // Calculate the data offset of the given source array within this storage.
    intptr_t offset = front_capacity_[nd-1] + shape_offset_vec[nd-1];
    intptr_t dim_offsets = 1;
    for(int i=nd-2; i>=0; --i)
    {
        dim_offsets *= (front_capacity_[i+1]+ shape_offset_vec[i+1] + src_shape[i+1] + (dst_shape[i+1] - shape_offset_vec[i+1] - src_shape[i+1]) + back_capacity_[i+1]);
        offset += (front_capacity_[i]+shape_offset_vec[i])*dim_offsets;
    }
    offset *= itemsize;

    // Calculate the strides of the given source array within this storage.
    std::vector<intptr_t> strides(nd, itemsize);
    for(int i=nd-2; i>=0; --i)
    {
        strides[i] = (front_capacity_[i+1] + shape_offset_vec[i+1] + src_shape[i+1] + (dst_shape[i+1] - shape_offset_vec[i+1] - src_shape[i+1]) + back_capacity_[i+1]) * strides[i+1];
    }

    // Create the destination array from this storage.
    bn::ndarray dst_arr = bn::from_data(bytearray_->get() + bytearray_data_offset_ + offset, dt_, src_shape, strides, /*owner=*/NULL, /*set_owndata_flag=*/false);

    // Now we just copy the src array to the dst array.
    if(! bn::copy_into(dst_arr, src_arr))
    {
        std::stringstream ss;
        ss << "Unable to copy the source array into the this ndarray storage!";
        throw MemoryError(ss.str());
    }
}

boost::shared_ptr<bytearray>
ndarray_storage::
create_bytearray(
    std::vector<intptr_t> const & shape
  , std::vector<intptr_t> const & front_capacity
  , std::vector<intptr_t> const & back_capacity
  , size_t const itemsize
)
{
    size_t const nd = shape.size();
    if(nd == 0)
    {
        throw ValueError(
            "The array must be at least 1-dimensional, i.e. "
            "len(shape) > 0!");
    }
    if(front_capacity.size() != nd ||
       back_capacity.size()  != nd
      )
    {
        throw ValueError(
            "The lengthes of shape, front_capacity and "
            "back_capacity must be equal!");
    }

    size_t capacity = 1;
    for(size_t i=0; i<nd; ++i)
    {
        const size_t cap_i = front_capacity[i] + shape[i] + back_capacity[i];
        capacity *= cap_i;
    }
    if(! (capacity > 0))
    {
        throw AssertionError(
            "The capacity is less or equal 0!");
    }

    return boost::shared_ptr<bytearray>(new bytearray(capacity, itemsize));
}

}// namespace detail
}// namespace ndhist
