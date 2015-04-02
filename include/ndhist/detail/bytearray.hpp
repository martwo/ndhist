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
#ifndef NDHIST_DETAIL_BYTEARRAY_H_INCLUDED
#define NDHIST_DETAIL_BYTEARRAY_H_INCLUDED

#include <cstring>
#include <iostream>

#include <boost/shared_ptr.hpp>

namespace ndhist {
namespace detail {

/**
 * @brief The bytearray class provides a very generic byte memory.
 */
class bytearray
{
  public:
    /**
     * @brief Allocates capacity*elsize number of bytes of new memory,
     *        initialized to zero. It returns the pointer to the new allocated
     *        memory after success and NULL otherwise. It uses the calloc C
     *        function, which is faster than malloc + memset for large chunks
     *        of memory allocation, due to OS specific memory management.
     *        (cf. http://stackoverflow.com/questions/2688466/why-mallocmemset-is-slower-than-calloc)
     */
    static
    char *
    calloc_data(size_t capacity, size_t elsize);

    /**
     * @brief Calls free on the given data.
     */
    static
    void free_data(char * data);

    /**
     * @brief Constructor for creating a new array of a certain capacity and
     *        element size.
     */
    bytearray(size_t capacity, size_t elsize)
      : data_(calloc_data(capacity, elsize))
      , bytesize_(capacity*elsize)
    {}

    /**
     * @brief Copy constructor for copying data from a given bytearray object.
     */
    bytearray(bytearray const & ba)
      : data_(calloc_data(ba.bytesize_, 1))
      , bytesize_(ba.bytesize_)
    {
        std::cout << "Copying bytearray ..." << std::flush;
        memcpy(data_, ba.data_, bytesize_);
        std::cout << "done." << std::endl<<std::flush;
    }

    virtual
    ~bytearray()
    {
        std::cout << "Destructing bytearray" << std::endl<<std::flush;
        if(data_)
        {
            free_data(data_);
        }
    }

    /** The pointer to the actual data byte array.
     */
    char * const data_;

    /** The size in bytes of this byte array.
     */
    size_t const bytesize_;

    /**
     * @brief Memsets all elements of the byte array to zero.
     */
    void
    clear();

    /**
     * @brief Returns the pointer to the beginning of the byte array.
     */
    char * get() const { return data_; }

    /**
     * @brief Creates a deepcopy of this bytearray on the heap.
     */
    boost::shared_ptr<bytearray>
    deepcopy()
    {
        // Use the copy constructor to make the actual copy.
        std::cout << "Deepcopy bytearray" << std::endl<<std::flush;
        return boost::shared_ptr<bytearray>(new bytearray(*this));
    }

  private:
    bytearray()
      : data_(NULL)
      , bytesize_(0)
    {}
};

}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_BYTEARRAY_H_INCLUDED
