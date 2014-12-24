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
 * This file contains preprocessor values for certain software limits which the
 * user might want to adjust before compiling the library.
 *
 */
#ifndef NDHIST_LIMITS_HPP_INCLUDED
#define NDHIST_LIMITS_HPP_INCLUDED 1

#include <ndhist/detail/limits.hpp>

/**
 * The maximal number of possible dimensions a histogram object can have. Note
 * that this limit is independent from the limit for the number of dimensions
 * possible for having the fill method available for a tuple of arrays (which is
 * dependent on the limits in BoostNumpy).
 * But as default value we set it to the same number (just for convienience).
 */
#ifndef NDHIST_LIMIT_MAX_ND
    #define NDHIST_LIMIT_MAX_ND \
        NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND
#endif

#endif // !NDHIST_LIMITS_HPP_INCLUDED
