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
 * This file contains preprocessor values for certain software limits. These
 * limits should be treated as hard-coded and depend on the limit values from
 * BoostNumpy.
 *
 */
#ifndef NDHIST_NDHIST_DETAIL_LIMITS_HPP_INCLUDED
#define NDHIST_NDHIST_DETAIL_LIMITS_HPP_INCLUDED 1

#include <boost/numpy/limits.hpp>

#define NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND \
        BOOST_NUMPY_LIMIT_INPUT_ARITY - 1

#endif // !NDHIST_NDHIST_DETAIL_LIMITS_HPP_INCLUDED
