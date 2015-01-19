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

#include <iostream>

#include <boost/noncopyable.hpp>

namespace ndhist {
namespace detail {

/**
 * @brief The bytearray class provides a very generic byte memory.
 */
class bytearray
  : public boost::noncopyable
{
  public:
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

    bytearray(size_t capacity, size_t elsize)
      : data_(calloc_data(capacity, elsize))
    {}

    virtual
    ~bytearray()
    {
        std::cout << "Destructing bytearray" << std::endl;
        if(data_)
        {
            free_data(data_);
        }
    }

    /** The pointer to the actual data byte array.
     */
    char * const data_;

    /**
     * @brief Returns the pointer to the beginning of the byte array.
     */
    char * get() { return data_; }

  private:
    bytearray()
      : data_(NULL)
    {}
};

}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_BYTEARRAY_H_INCLUDED
