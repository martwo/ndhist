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

#include <boost/function.hpp>
#include <boost/python.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/detail/ndarray_storage.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

struct ndhist;

namespace detail {

struct AxisData;

struct Axis
{
    boost::function<intptr_t (boost::shared_ptr<AxisData>, bp::object const &)> get_bin_index_fct;
    boost::shared_ptr<AxisData> data_;
};

struct AxisData
{};

struct GenericAxisData : AxisData
{
    boost::shared_ptr<detail::ndarray_storage> storage_;
    std::vector<bn::ndarray> arr_;
};

struct GenericAxis : Axis
{
    GenericAxis(::ndhist::ndhist * h, bn::ndarray const & edges, intptr_t front_capacity=0, intptr_t back_capacity=0)
    {
        get_bin_index_fct = &GenericAxis::get_bin_index;
        data_ = boost::shared_ptr<GenericAxisData>(new GenericAxisData());
        GenericAxisData & ddata = *static_cast<GenericAxisData*>(data_.get());
        intptr_t const nbins = edges.get_size();
        std::vector<intptr_t> shape(1, nbins);
        std::vector<intptr_t> front_capacity_vec(1, front_capacity);
        std::vector<intptr_t> back_capacity_vec(1, back_capacity);
        ddata.storage_ = boost::shared_ptr<detail::ndarray_storage>(
            new detail::ndarray_storage(shape, front_capacity_vec, back_capacity_vec, edges.get_dtype()));
        // Copy the data from the user provided edge array to the storage array.

        bp::object owner(bp::ptr(h));
        ddata.arr_.push_back(ddata.storage_->ConstructNDArray(&owner));
        if(!bn::copy_into(ddata.arr_[0], edges))
        {
            // FIXME: Get the error string from the already set BP error.
            throw MemoryError(
               "Could not copy edge array into internal storage!");
        }
    }

    static
    intptr_t
    get_bin_index(boost::shared_ptr<AxisData> axisdata, bp::object const & value)
    {
        GenericAxisData & data = *static_cast<GenericAxisData *>(axisdata.get());
        // We know that edges is 1-dimensional by construction and the edges are
        // ordered ascedently.
        intptr_t const N = data.arr_[0].get_size();
        for(intptr_t i=0; i<N; ++i)
        {
            bp::object lower_edge = data.arr_[0].get_item<bp::object, intptr_t>(i);
            PyTypeObject* lower_edge_type = (PyTypeObject*)PyObject_Type(lower_edge.ptr());
            if((! PyObject_TypeCheck(value.ptr(), lower_edge_type)) &&
            (! (bn::is_any_scalar(value) && bn::is_any_scalar(lower_edge)) )
            )
            {
                Py_DECREF(lower_edge_type);
                std::stringstream ss;
                ss << "The value for axis " << i+1 << " must be a subclass of the "
                << "edges objects of axis "<< i+1 << " of the same type! "
                << "Otherwise comparison operators might be ill-defined.";
                throw TypeError(ss.str());
            }
            Py_DECREF(lower_edge_type);

            if(lower_edge > value)
            {
                return i-1;
            }
        }

        return -2;
    }
};




}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_AXIS_HPP_INCLUDED
