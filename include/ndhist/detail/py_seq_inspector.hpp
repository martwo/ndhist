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
#ifndef NDHIST_DETAIL_PY_SEQ_INSPECTOR_HPP_INCLUDED
#define NDHIST_DETAIL_PY_SEQ_INSPECTOR_HPP_INCLUDED

#include <boost/python.hpp>

#include <ndhist/detail/utils.hpp>

namespace bp = boost::python;

namespace ndhist {
namespace detail {
namespace py {

struct seq_inspector
{
    bp::object const & seq_;

    seq_inspector(bp::object const & seq)
      : seq_(seq)
    {}

    /**
     * @brief Checks if the sequence has only one (or zero) objects of the given
     *     Python type.
     */
    inline
    bool
    has_unique_object_of_type(PyTypeObject & type_obj)
    {
        bool found_once = false;
        size_t const n = bp::len(seq_);
        for(size_t i=0; i<n; ++i)
        {
            bp::object item = bp::extract<bp::object>(seq_[i]);
            if(py::is_object_of_type(item, type_obj))
            {
                if(found_once)
                {
                    return false;
                }
                else
                {
                    found_once = true;
                }
            }
        }

        return true;
    }
};

}//namespace py
}//namespace detail
}//namespace ndhist

#endif // !NDHIST_DETAIL_PY_SEQ_INSPECTOR_HPP_INCLUDED
