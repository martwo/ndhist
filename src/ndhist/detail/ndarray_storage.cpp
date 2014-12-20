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
extend_axis(intptr_t axis, intptr_t n_elements)
{
    std::cout << "extend axis " << axis << "by " << n_elements << std::endl;
    if(n_elements == 0) return;
    if(n_elements < 0)
    {
        n_elements = -n_elements;

        if(front_capacity_[axis] - n_elements >= 0)
        {
            // The capacity is still sufficient for the extention. No need for
            // memory reallocation.
            std::cout << "Increment shape[axis] "<< shape_[axis] << "by "<<n_elements<<std::endl;
            shape_[axis] += n_elements;
            front_capacity_[axis] -= n_elements;
        }
        else
        {
            // The capacity is not sufficient. Reallocate the memory with a
            // bigger size.
            // FIXME
            throw MemoryError("The reallocation (front) is not implemented yet.");
        }
    }
    else // n_elements > 0
    {
        if(back_capacity_[axis] - n_elements >= 0)
        {
            // The capacity is still sufficient for the extention. No need for
            // memory reallocation.
            shape_[axis] += n_elements;
            back_capacity_[axis] -= n_elements;
        }
        else
        {
            // The capacity is not sufficient. Reallocate the memory with a
            // bigger size.
            // FIXME
            throw MemoryError("The reallocation (back) is not implemented yet.");
        }
    }
}

//______________________________________________________________________________
void
ndarray_storage::
Calloc(size_t capacity, size_t elsize)
{
    std::cout << "Calloc " << capacity << " elements of size " << elsize << std::endl;
    data_ = (char*)calloc(capacity, elsize);
    if(data_ == NULL)
    {
        throw MemoryError("Unable to allocate memory!");
    }
}

//______________________________________________________________________________
void
ndarray_storage::
Free()
{
    free(data_);
    data_ = NULL;
}

}// namespace detail
}// namespace ndhist
