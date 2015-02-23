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
#ifndef NDHIST_ERROR_HPP_INCLUDED
#define NDHIST_ERROR_HPP_INCLUDED 1

#include <ndhist/detail/error.hpp>

namespace ndhist {

typedef detail::error<detail::AssertionErrorType> AssertionError;
typedef detail::error<detail::IndexErrorType>     IndexError;
typedef detail::error<detail::MemoryErrorType>    MemoryError;
typedef detail::error<detail::RuntimeErrorType>   RuntimeError;
typedef detail::error<detail::TypeErrorType>      TypeError;
typedef detail::error<detail::ValueErrorType>     ValueError;

}// namespace ndhist

#endif // ! NDHIST_ERROR_HPP_INCLUDED
