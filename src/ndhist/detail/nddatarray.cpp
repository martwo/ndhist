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

#include <ndhist/detail/nddatarray.hpp>

namespace ndhist {
namespace detail {

//______________________________________________________________________________
void
nddatarray::
Calloc(size_t capacity, size_t elsize)
{
    data_ = (char*)calloc(capacity, elsize);
    if(data_ == NULL)
    {
        throw error("Unable to allocate memory!");
    }
}

//______________________________________________________________________________
void
nddatarray::
Free()
{
    free(data_);
    data_ = NULL;
}

}// namespace detail
}// namespace ndhist
