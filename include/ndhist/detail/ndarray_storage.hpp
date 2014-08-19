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
#ifndef NDHIST_DETAIL_NDARRAY_STORAGE_H_INCLUDED
#define NDHIST_DETAIL_NDARRAY_STORAGE_H_INCLUDED 1

#include <vector>

#include <boost/numpy/dtype.hpp>

#include <ndhist/error.hpp>

namespace bn = boost::numpy;

namespace ndhist {
namespace detail {

/**
 * The ndarray_storage class provides storage management functionalities for
 * a ndarray object through a generic contigious byte array. This byte array
 * can be accessed via a ndarray object which can be build around this storage.
 * The storage can be bigger than the ndarray accesses. This allows to add
 * additional (hidden) capacity to arrays, which can be used for growing the
 * ndarray without memory re-allocation.
 *
 */
class ndarray_storage
{
  public:
    ndarray_storage(
        std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , bn::dtype const & dt
    )
      : shape_(shape)
      , front_capacity_(front_capacity)
      , back_capacity_(back_capacity)
      , dt_(dt)
      , data_(NULL)
    {
        const size_t nd = shape_.size();
        if(! (front_capacity_.size() == nd &&
              back_capacity_.size()  == nd)
          )
        {
            throw ValueError(
                "The lengthes of shape, front_capacity and "
                "back_capacity must be equal!");
        }
        if(nd == 0)
        {
            throw ValueError(
                "The array must be at least 1-dimensional, i.e. "
                "len(shape) > 0!");
        }
        size_t capacity = 1;
        for(size_t i=0; i<nd; ++i)
        {
            const size_t cap_i = front_capacity_[i] + shape_[i] + back_capacity_[i];
            capacity *= cap_i;
        }
        if(! (capacity > 0))
        {
            throw AssertionError(
                "The capacity is less or equal 0!");
        }
        Calloc(capacity, dt_.get_itemsize());
    }

    virtual
    ~ndarray_storage()
    {
        if(data_)
        {
            Free();
        }
    }

  protected:
    /**
     * Allocates capacity*elsize number of bytes of new memory, initialized to
     * zero. It returns "true" after success and "false" otherwise. It uses the
     * calloc C function, which is faster than malloc + memset for large chunks
     * of memory allocation, due to OS specific memory management.
     * (cf. http://stackoverflow.com/questions/2688466/why-mallocmemset-is-slower-than-calloc)
     */
    void Calloc(size_t capacity, size_t elsize);

    /**
     * Calls free on data_ and sets data_ to NULL.
     */
    void Free();


  private:
    /** The shape defines the number of dimensions and how many elements each
     *  dimension has.
     */
    std::vector<intptr_t> shape_;

    /** The additional front and back capacities define how many additional
     *  elements each dimension has.
     */
    std::vector<intptr_t> front_capacity_;
    std::vector<intptr_t> back_capacity_;

    /** The numpy data type object, defining the element size in bytes.
     */
    bn::dtype dt_;

    /** The actual data storage.
     */
    char* data_;
};

}// namespace detail
}// namespace ndhist

#endif // ! NDHIST_DETAIL_NDARRAY_STORAGE_H_INCLUDED
