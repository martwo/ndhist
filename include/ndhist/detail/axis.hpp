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
#ifndef NDHIST_DETAIL_AXIS_HPP_INCLUDED
#define NDHIST_DETAIL_AXIS_HPP_INCLUDED 1

namespace ndhist {
namespace detail {
namespace axis {

// intptr_t const UNDERFLOW_INDEX = -1;
// intptr_t const OVERFLOW_INDEX  = -2;

enum axis_flags_t
{
    FLAGS_FIXED_INDEX    = -4,
    FLAGS_FLOATING_INDEX = -8
};

}// namespace axis
}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_AXIS_HPP_INCLUDED
