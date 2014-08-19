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

#include <ndhist/detail/ndarray_storage.hpp>

namespace ndhist {
namespace detail {

//______________________________________________________________________________
void
ndarray_storage::
Calloc(size_t capacity, size_t elsize)
{
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
