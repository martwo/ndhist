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

#include <iostream>
#include <vector>

#include <boost/python.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>

#include <ndhist/error.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace detail {

/**
 * The ndarray_storage class provides storage management functionalities for
 * a ndarray object through a generic contigious byte array. This byte array
 * can be accessed via a ndarray object which can be build around this storage.
 * The storage can be bigger than what the ndarray accesses. This allows to add
 * additional (hidden) capacity to arrays, which can be used for growing the
 * ndarray without memory re-allocation.
 *
 */
class ndarray_storage
  : public boost::noncopyable
{
  public:
    ndarray_storage(
        std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , boost::numpy::dtype   const & dt
    )
      : shape_(shape)
      , front_capacity_(front_capacity)
      , back_capacity_(back_capacity)
      , dt_(dt)
      , data_(NULL)
    {
        data_ = create_array_data(shape_, front_capacity_, back_capacity_, dt_);
    }

    virtual
    ~ndarray_storage()
    {
        std::cout << "Destructing ndarray_storage" << std::endl;
        if(data_)
        {
            free_data(data_);
            data_ = NULL;
        }
    }

    int get_nd() const { return shape_.size(); }

    /** Calculates the offset of the data pointer needed for a ndarray wrapping
     *  this ndarray storage.
     */
    intptr_t CalcDataOffset() const;

    /** Calculates the data strides for a ndarray wrapping this ndarray storage.
     */
    std::vector<intptr_t> CalcDataStrides() const;

    /** Constructs a boost::numpy::ndarray object wrapping this ndarray storage
     *  with the correct layout, i.e. strides.
     */
    boost::numpy::ndarray
    ConstructNDArray(boost::python::object const * data_owner=NULL);


    /** Extends the memory of this ndarray storage by at least the given number
     *  of elements for each axis. The n_elements_vec argument must hold the
     *  number of extra elements (can be zero) for each axis. Negative numbers
     *  indicate an extension to the left and positive numbers to the right of
     *  the axis. If all the axes
     *  have still enough capacity to hold the new elements, no reallocation of
     *  memory is performed. Otherwise a complete new junk of memory is
     *  allocated that can fit the extended array plus the specified extra font
     *  and back capacity. The data from the old array is copied to the new
     *  array.
     */
    void
    extend_axes(
        std::vector<intptr_t> const & n_elements_vec
      , std::vector<intptr_t> const & max_fcap_vec
      , std::vector<intptr_t> const & max_bcap_vec
      , bp::object const * data_owner
    );

    /** Allocates capacity*elsize number of bytes of new memory, initialized to
     *  zero. It returns the pointer to the new allocated memory after success
     *  and NULL otherwise. It uses the
     *  calloc C function, which is faster than malloc + memset for large chunks
     *  of memory allocation, due to OS specific memory management.
     *  (cf. http://stackoverflow.com/questions/2688466/why-mallocmemset-is-slower-than-calloc)
     */
    static
    char *
    calloc_data(size_t capacity, size_t elsize);

    /** Calls free on the given data.
     */
    static
    void free_data(char * data);

  protected:
    /** Creates (i.e. allocates) data for the given array layout. It returns
     *  the pointer to the new allocated memory.
     */
    static
    char *
    create_array_data(
        std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , boost::numpy::dtype   const & dt
    );

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
    boost::numpy::dtype dt_;

    /** The pointer to the actual data storage.
     */
    char * data_;
};

}// namespace detail
}// namespace ndhist

#endif // ! NDHIST_DETAIL_NDARRAY_STORAGE_H_INCLUDED
