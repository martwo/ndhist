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
#ifndef NDHIST_DETAIL_UTILS_HPP_INCLUDED
#define NDHIST_DETAIL_UTILS_HPP_INCLUDED 1

namespace ndhist {
namespace detail {

namespace py {

inline
bool
are_same_type_objects(bp::object const & obj1, bp::object const & obj2)
{
    bp::object obj2_type(bp::handle<>(bp::borrowed(bp::downcast<PyTypeObject>(PyObject_Type(obj2.ptr())))));
    return PyObject_TypeCheck(obj1.ptr(), (PyTypeObject*)obj2_type.ptr());
}

}// namespace py

}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_UTILS_HPP_INCLUDED
