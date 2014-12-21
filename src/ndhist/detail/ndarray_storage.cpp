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
CalcDataOffset() const
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

    return offset;
}

//______________________________________________________________________________
std::vector<intptr_t>
ndarray_storage::
CalcDataStrides() const
{
    const int itemsize = dt_.get_itemsize();
    std::vector<intptr_t> strides(shape_.size(), itemsize);
    std::cout << "strides["<<strides.size()-1<<"] = " << strides[strides.size()-1] << std::endl;
    for(int i=strides.size()-2; i>=0; --i)
    {
        strides[i] = ((front_capacity_[i+1] + shape_[i+1] + back_capacity_[i+1]) * strides[i+1]/itemsize)*itemsize;
        std::cout << "strides["<<i<<"] = " << strides[i] << std::endl;
    }

    return strides;
}

//______________________________________________________________________________
bn::ndarray
ndarray_storage::
ConstructNDArray(bp::object const * data_owner)
{
    return bn::from_data(data_+CalcDataOffset(), dt_, shape_, CalcDataStrides(), data_owner);
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
    if(f_n_elements_vec.size() != nd ||
       b_n_elements_vec.size() != nd
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

    std::cout << "ndarray_storage::extend_axes" <<std::endl<<std::flush;

    if(reallocate)
    {
        std::vector<intptr_t> old_shape = shape_;
        std::vector<intptr_t> old_fcap = front_capacity_;
        std::vector<intptr_t> old_bcap = back_capacity_;
        // TODO: f_diffs is f_n_elements_vec and b_diffs is b_n_elements_vec.
        //       Get rid of f_diffs and b_diffs.
        std::vector<intptr_t> f_diffs(nd, 0);
        std::vector<intptr_t> b_diffs(nd, 0);
        for(int i=0; i<nd; ++i)
        {
            intptr_t const f_n_elements = f_n_elements_vec[i];
            intptr_t const b_n_elements = b_n_elements_vec[i];
            if(f_n_elements > 0)
            {
                old_shape[i] -= f_n_elements;
                old_fcap[i] += f_n_elements;
                f_diffs[i] = f_n_elements;
                front_capacity_[i] = max_fcap_vec[i];
            }
            if(b_n_elements > 0)
            {
                old_shape[i] -= b_n_elements;
                old_bcap[i] += b_n_elements;
                b_diffs[i] = b_n_elements;
                back_capacity_[i] = max_bcap_vec[i];
            }
        }

        std::cout << "ndarray_storage::extend_axes: reallocate ++++++++++++++++" <<std::endl<<std::flush;
        // At this point shape_, front_capacity_ and back_capacity_ have the
        // right numbers for the new array.
        // Allocate the new array memory.
        char * new_data = create_array_data(shape_, front_capacity_, back_capacity_, dt_);

        // Copy the data from the old memory to the new one. We do this by
        // creating two ndarrays having the same layout. The first is the old
        // array layout on the new memory and the second is the old array on the
        // old memory. Then we just use the copy_into function to copy the data.
        // We assume that the numpy people implemented the PyArray_CopyInto
        // C-API function efficient enough ;)
        const int itemsize = dt_.get_itemsize();

        // Calculate data offset of the old array inside the new memory.
        intptr_t new_offset = front_capacity_[nd-1] + f_diffs[nd-1];
        intptr_t new_dim_offsets = 1;
        for(int i=nd-2; i>=0; --i)
        {
            new_dim_offsets *= (front_capacity_[i+1]+f_diffs[i+1] + old_shape[i+1] + back_capacity_[i+1]+b_diffs[i+1]);
            new_offset += (front_capacity_[i]+f_diffs[i])*new_dim_offsets;
        }
        new_offset *= itemsize;

        // Calculate the strides of the old array inside the new memory.
        std::vector<intptr_t> new_strides(nd, itemsize);
        for(int i=nd-2; i>=0; --i)
        {
            new_strides[i] = ((front_capacity_[i+1]+f_diffs[i+1] + old_shape[i+1] + back_capacity_[i+1]+b_diffs[i+1]) * new_strides[i+1]/itemsize)*itemsize;
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
}

//______________________________________________________________________________
char *
ndarray_storage::
calloc_data(size_t capacity, size_t elsize)
{
    std::cout << "Calloc " << capacity << " elements of size " << elsize << std::endl;
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
  , boost::numpy::dtype   const & dt
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
    return calloc_data(capacity, dt.get_itemsize());
}

}// namespace detail
}// namespace ndhist
