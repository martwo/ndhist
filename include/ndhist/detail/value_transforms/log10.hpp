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
#ifndef NDHIST_DETAIL_VALUE_TRANSFORMS_LOG10_HPP_INCLUDED
#define NDHIST_DETAIL_VALUE_TRANSFORMS_LOG10_HPP_INCLUDED 1

#include <cmath>

namespace ndhist {
namespace detail {
namespace value_transforms {

template <typename ValueType>
class log10
{
  public:
    /** Transforms a given value to its log10 representation.
     */
    inline
    static
    ValueType
    transform(ValueType const value)
    {
        return std::log10(value);
    }

    /** Back-transforms a given value from its log10 representation to actual
     *  value.
     */
    inline
    static
    ValueType
    back_transform(ValueType const value)
    {
        return std::pow(10, value);
    }
};

}// namespace value_transforms
}// namespace detail
}// namespace ndhist

#endif // ! NDHIST_DETAIL_VALUE_TRANSFORMS_IDENTITY_HPP_INCLUDED
