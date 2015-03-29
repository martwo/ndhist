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
#ifndef NDHIST_TYPE_SUPPORT_HPP_INCLUDED
#define NDHIST_TYPE_SUPPORT_HPP_INCLUDED

#include <boost/preprocessor/seq/pop_back.hpp>

/** Define the list of supported axis value types.
 */
#ifndef NDHIST_TYPE_SUPPORT_AXIS_VALUE_TYPES
#define NDHIST_TYPE_SUPPORT_AXIS_VALUE_TYPES \
    (float)\
    (double)\
/*    (boost::python::object)*/
#endif

/** Define the list of supported weight value types (including
 *  boost::python::object).
 *  Note: The last element MUST BE boost::python::object !!!
 */
#ifndef NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES
#define NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES \
    (bool)\
    (int8_t)\
    (uint8_t)\
    (int16_t)\
    (uint16_t)\
    (int32_t)\
    (uint32_t)\
    (int64_t)\
    (uint64_t)\
    (float)\
    (double)\
    (boost::python::object)
#endif

/** Define the list of supported weight value types (excluding
 *  boost::python::object).
 */
#define NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES_WITHOUT_OBJECT \
    BOOST_PP_SEQ_POP_BACK(NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)

#endif // NDHIST_TYPE_SUPPORT_HPP_INCLUDED
