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
    std::vector<intptr_t> const & n_elements_vec
  , std::vector<intptr_t> const & min_fcap_vec
  , std::vector<intptr_t> const & min_bcap_vec
)
{
    int const nd = this->get_nd();
    if(n_elements_vec.size() != nd)
    {
        std::stringstream ss;
        ss << "The vector holding the number of new elements is "
           << n_elements_vec.size() <<", but must be " << nd << "!";
        throw AssertionError(ss.str());
    }

    // First check if a memory reallocation is actually required.
    bool reallocate = false;
    for(int axis=0; axis<=nd; ++axis)
    {
        intptr_t const n_elements = n_elements_vec[axis];
        if(n_elements == 0) continue;
        if(n_elements < 0)
        {
            if(front_capacity_[axis] + n_elements < 0)
            {
                reallocate = true;
            }
            shape_[axis] -= n_elements;
            front_capacity_[axis] += n_elements;
        }
        else // // n_elements > 0
        {
            if(back_capacity_[axis] - n_elements < 0)
            {
                reallocate = true;
            }
            shape_[axis] += n_elements;
            back_capacity_[axis] -= n_elements;


        }
    }

    std::cout << "ndarray_storage::extend_axes" <<std::endl<<std::flush;

    if(reallocate)
    {
        // The negative values in front_capacity_ and back_capacity_ specify
        // the number of extra elements required to allocate for each axis.
        throw MemoryError("The reallocation (front & back) is not implemented yet.");
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
