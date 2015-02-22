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

// inline
// bool
// are_same_type_objects(bp::object const & obj1, bp::object const & obj2)
// {
//     bp::object obj2_type(bp::handle<>(bp::borrowed(bp::downcast<PyTypeObject>(PyObject_Type(obj2.ptr())))));
//     return PyObject_TypeCheck(obj1.ptr(), (PyTypeObject*)obj2_type.ptr());
// }

inline
bool
is_object_of_type(
    bp::object const & obj
  , PyTypeObject & type_obj0
)
{
    if(PyObject_TypeCheck(obj.ptr(), &type_obj0))
    {
        return true;
    }
    return false;
}

inline
bool
is_object_of_type(
    bp::object const & obj
  , PyTypeObject & type_obj0
  , PyTypeObject & type_obj1
)
{
    if(   PyObject_TypeCheck(obj.ptr(), &type_obj0)
       || PyObject_TypeCheck(obj.ptr(), &type_obj1)
      )
    {
        return true;
    }
    return false;
}

inline
bool
is_object_of_type(
    bp::object const & obj
  , PyTypeObject & type_obj0
  , PyTypeObject & type_obj1
  , PyTypeObject & type_obj2
)
{
    if(   PyObject_TypeCheck(obj.ptr(), &type_obj0)
       || PyObject_TypeCheck(obj.ptr(), &type_obj1)
       || PyObject_TypeCheck(obj.ptr(), &type_obj2)
      )
    {
        return true;
    }
    return false;
}

}// namespace py

}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_UTILS_HPP_INCLUDED
