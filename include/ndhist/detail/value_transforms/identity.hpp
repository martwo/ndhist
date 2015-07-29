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
#ifndef NDHIST_DETAIL_VALUE_TRANSFORMS_IDENTITY_HPP_INCLUDED
#define NDHIST_DETAIL_VALUE_TRANSFORMS_IDENTITY_HPP_INCLUDED 1

namespace ndhist {
namespace detail {
namespace value_transforms {

template <typename ValueType>
class identity
{
  public:
    /** Transforms a given value to itself.
     */
    inline
    static
    ValueType
    transform(ValueType const value)
    {
        return value;
    }

    /** Back-transforms a given value to itself.
     */
    inline
    static
    ValueType
    back_transform(ValueType const value)
    {
        return value;
    }
};

}// namespace value_transforms
}// namespace detail
}// namespace ndhist

#endif // ! NDHIST_DETAIL_VALUE_TRANSFORMS_IDENTITY_HPP_INCLUDED
