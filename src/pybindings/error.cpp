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
#include <boost/preprocessor/cat.hpp>
#include <boost/python.hpp>

#include <ndhist/error.hpp>

namespace bp = boost::python;

namespace ndhist {

template <class ExcType>
static
void translate(ExcType const & e);

#define NDHIST_ERROR_TRANSLATE(exctype)                             \
    template <>                                                     \
    void translate<exctype>(exctype const & e)                      \
    {                                                               \
        /* Use the Python 'C' API to set up an exception object. */ \
        PyErr_SetString( BOOST_PP_CAT(PyExc_, exctype), e.what() ); \
    }

NDHIST_ERROR_TRANSLATE(AssertionError)
NDHIST_ERROR_TRANSLATE(IndexError)
NDHIST_ERROR_TRANSLATE(MemoryError)
NDHIST_ERROR_TRANSLATE(ValueError)

#undef NDHIST_ERROR_TRANSLATE

void register_error_types()
{
    bp::register_exception_translator<AssertionError>(&translate<AssertionError>);
    bp::register_exception_translator<IndexError>    (&translate<IndexError>);
    bp::register_exception_translator<MemoryError>   (&translate<MemoryError>);
    bp::register_exception_translator<ValueError>    (&translate<ValueError>);
}

}// namespace ndhist
