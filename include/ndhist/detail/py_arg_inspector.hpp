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
#ifndef NDHIST_DETAIL_PY_ARG_INSPECTOR_HPP_INCLUDED
#define NDHIST_DETAIL_PY_ARG_INSPECTOR_HPP_INCLUDED

#include <boost/python.hpp>

#include <ndhist/detail/utils.hpp>

namespace bp = boost::python;

namespace ndhist {
namespace detail {
namespace py {

struct arg_inspector
{
    bp::object const & arg_;

    arg_inspector(bp::object const & arg)
      : arg_(arg)
    {}

    /**
     * @brief Checks if the argument is an object of one of the given Python
     *     types.
     */
    inline
    bool
    is_of_type(
        PyTypeObject & type_obj0
    )
    {
        return is_object_of_type(arg_, type_obj0);
    }

    inline
    bool
    is_of_type(
        PyTypeObject & type_obj0
      , PyTypeObject & type_obj1
    )
    {
        return is_object_of_type(arg_, type_obj0, type_obj1);
    }

    inline
    bool
    is_of_type(
        PyTypeObject & type_obj0
      , PyTypeObject & type_obj1
      , PyTypeObject & type_obj2
    )
    {
        return is_object_of_type(arg_, type_obj0, type_obj1, type_obj2);
    }

    inline
    bool
    is_tuple()
    {
        return is_of_type(PyTuple_Type);
    }

    /**
     * @brief Checks if the argument is a tuple of objects of one of the given
     *     Python types (Mixing of types in the tuple is allowed).
     */
    inline
    bool
    is_tuple_of(
        PyTypeObject & type_obj0
      , PyTypeObject & type_obj1
      , PyTypeObject & type_obj2
    )
    {
        if(! is_of_type(PyTuple_Type))
        {
            return false;
        }

        // Loop over the tuple and check each element.
        size_t const n = bp::len(arg_);
        for(size_t i=0; i<n; ++i)
        {
            bp::object item = bp::extract<bp::object>(arg_[i]);
            if(! is_object_of_type(item, type_obj0, type_obj1, type_obj2))
            {
                return false;
            }
        }

        return true;
    }

    inline
    bool
    is_list_of(
        PyTypeObject & type_obj0
    )
    {
        if(! is_of_type(PyList_Type))
        {
            return false;
        }

        // Loop over the list and check each element.
        size_t const n = bp::len(arg_);
        for(size_t i=0; i<n; ++i)
        {
            bp::object item = bp::extract<bp::object>(arg_[i]);
            if(! is_object_of_type(item, type_obj0))
            {
                return false;
            }
        }

        return true;
    }

    inline
    bool
    is_list_of(
        PyTypeObject & type_obj0
      , PyTypeObject & type_obj1
      , PyTypeObject & type_obj2
    )
    {
        if(! is_of_type(PyList_Type))
        {
            return false;
        }

        // Loop over the list and check each element.
        size_t const n = bp::len(arg_);
        for(size_t i=0; i<n; ++i)
        {
            bp::object item = bp::extract<bp::object>(arg_[i]);
            if(! is_object_of_type(item, type_obj0, type_obj1, type_obj2))
            {
                return false;
            }
        }

        return true;
    }
};

}//namespace py
}//namespace detail
}//namespace ndhist

#endif // !NDHIST_DETAIL_PY_ARG_INSPECTOR_HPP_INCLUDED
