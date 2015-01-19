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
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <sstream>

#include <ndhist/detail/bytearray.hpp>
#include <ndhist/error.hpp>

namespace ndhist {
namespace detail {

char *
bytearray::
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

void
bytearray::
free_data(char * data)
{
    free(data);
}

}// namespace detail
}// namespace ndhist
