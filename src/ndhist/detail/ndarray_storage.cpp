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

#include <ndhist/detail/ndarray_storage.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace detail {

//______________________________________________________________________________
intptr_t
ndarray_storage::
CalcDataOffset(size_t sub_item_byte_offset) const
{
    const int nd = shape_.size();
    intptr_t offset = front_capacity_[nd-1];
    intptr_t dim_offsets = 1;
    for(int i=nd-2; i>=0; --i)
    {
        dim_offsets *= (front_capacity_[i+1] + shape_[i+1] + back_capacity_[i+1]);
        offset += front_capacity_[i]*dim_offsets;
    }
    offset *= dt_.get_itemsize();

    return offset + sub_item_byte_offset;
}

//______________________________________________________________________________
void
ndarray_storage::
calc_data_strides(std::vector<intptr_t> & strides) const
{
    size_t const nd = shape_.size();
    int const itemsize = dt_.get_itemsize();
    strides[nd-1] = itemsize;
    for(intptr_t i=nd-2; i>=0; --i)
    {
        strides[i] = ((front_capacity_[i+1] + shape_[i+1] + back_capacity_[i+1]) * strides[i+1]/itemsize)*itemsize;
    }
}

//______________________________________________________________________________
bn::ndarray
ndarray_storage::
ConstructNDArray(bn::dtype const & dt, size_t field_idx, bp::object const * data_owner)
{
    //std::cout << "ConstructNDArray for field_idx "<< field_idx << std::endl;
    size_t sub_item_byte_offset = 0;
    if(field_idx > 0)
    {
        sub_item_byte_offset = dt_.get_fields_byte_offsets()[field_idx];
    }
    //std::cout << "sub_item_byte_offset = "<<sub_item_byte_offset<<std::endl;
    return bn::from_data(data_+CalcDataOffset(sub_item_byte_offset), dt, shape_, get_data_strides_vector(), data_owner);
}

//______________________________________________________________________________
void
ndarray_storage::
extend_axes(
    std::vector<intptr_t> const & f_n_elements_vec
  , std::vector<intptr_t> const & b_n_elements_vec
  , std::vector<intptr_t> const & max_fcap_vec
  , std::vector<intptr_t> const & max_bcap_vec
  , bp::object const * data_owner
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

    //std::cout << "ndarray_storage::extend_axes" <<std::endl<<std::flush;

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

        // At this point shape_, front_capacity_ and back_capacity_ have the
        // right numbers for the new array.
        // Allocate the new array memory.
        char * new_data = create_array_data(shape_, front_capacity_, back_capacity_, dt_.get_itemsize());

        // Copy the data from the old memory to the new one. We do this by
        // creating two ndarrays having the same layout. The first is the old
        // array layout on the new memory and the second is the old array on the
        // old memory. Then we just use the copy_into function to copy the data.
        // We assume that the numpy people implemented the PyArray_CopyInto
        // C-API function efficient enough ;)
        const int itemsize = dt_.get_itemsize();

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
            new_strides[i] = ((front_capacity_[i+1]+f_n_elements_vec[i+1] + old_shape[i+1] + back_capacity_[i+1]+b_n_elements_vec[i+1]) * new_strides[i+1]/itemsize)*itemsize;
        }

        // Create the old array from the new memory.
        bn::ndarray new_data_arr = bn::from_data(new_data+new_offset, dt_, old_shape, new_strides, data_owner);

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
            old_strides[i] = ((old_fcap[i+1] + old_shape[i+1] + old_bcap[i+1]) * old_strides[i+1]/itemsize)*itemsize;
        }

        // Create the old array from the old memory.
        bn::ndarray old_data_arr = bn::from_data(data_+old_offset, dt_, old_shape, old_strides, data_owner);

        // Now we just copy the old data to the new data.
        if(! bn::copy_into(new_data_arr, old_data_arr))
        {
            throw MemoryError("Unable to copy the old data from the old memory "
                              "into the new memory!");
        }

        // Deallocate the old data and assign the new data to this storage.
        free_data(data_);
        data_ = new_data;
    }

    // Recalculate the data strides.
    calc_data_strides(data_strides_);
}

void
ndarray_storage::
copy_from(
    bn::ndarray const & src_arr
  , std::vector<intptr_t> const & shape_offset_vec
)
{
    int const nd = this->get_nd();
    int const itemsize = dt_.get_itemsize();

    std::vector<intptr_t> const & dst_shape = this->get_shape_vector();
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
        strides[i] = ((front_capacity_[i+1]+ shape_offset_vec[i+1] + src_shape[i+1] + (dst_shape[i+1] - shape_offset_vec[i+1] - src_shape[i+1]) + back_capacity_[i+1]) * strides[i+1]/itemsize)*itemsize;
    }

    // Create the destination array from this storage.
    bp::object data_owner; // We use the None object as an owner proxy.
    bn::ndarray dst_arr = bn::from_data(data_+offset, dt_, src_shape, strides, &data_owner);

    // Now we just copy the src array to the dst array.
    if(! bn::copy_into(dst_arr, src_arr))
    {
        throw MemoryError("Unable to copy the source array into the this "
                          "ndarray storage!");
    }
}

//______________________________________________________________________________
char *
ndarray_storage::
calloc_data(size_t capacity, size_t elsize)
{
    //std::cout << "Calloc " << capacity << " elements of size " << elsize << std::endl;
    char * data = (char*)calloc(capacity, elsize);
    if(data == NULL)
    {
        std::stringstream ss;
        ss << "Unable to allocate " << capacity << " elements of size "
           << elsize <<" of memory!";
        throw MemoryError(ss.str());
    }
    return data;
}

//______________________________________________________________________________
void
ndarray_storage::
free_data(char * data)
{
    free(data);
}

//______________________________________________________________________________
char *
ndarray_storage::
create_array_data(
    std::vector<intptr_t> const & shape
  , std::vector<intptr_t> const & front_capacity
  , std::vector<intptr_t> const & back_capacity
  , size_t itemsize
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
    return calloc_data(capacity, itemsize);
}

}// namespace detail
}// namespace ndhist
