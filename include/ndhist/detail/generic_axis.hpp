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
#ifndef NDHIST_DETAIL_GENERIC_AXIS_HPP_INCLUDED
#define NDHIST_DETAIL_GENERIC_AXIS_HPP_INCLUDED 1

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/flat_iterator.hpp>

#include <ndhist/detail/axis.hpp>
#include <ndhist/detail/ndarray_storage.hpp>
#include <ndhist/ndhist.hpp>

namespace ndhist {
namespace detail {

template<typename AxisValueType>
struct GenericAxisData : AxisData
{
    boost::shared_ptr<ndarray_storage> storage_;
    bp::object arr_;
    bn::flat_iterator<AxisValueType> iter_;
    bn::flat_iterator<AxisValueType> iter_end_;
};

template <typename AxisValueType>
struct GenericAxis
  : Axis
{
    GenericAxis(::ndhist::ndhist * h, bn::ndarray const & edges, intptr_t front_capacity=0, intptr_t back_capacity=0)
    {
        // Set up the axis's function pointers.
        get_bin_index_fct     = &GenericAxis<AxisValueType>::get_bin_index;
        get_edges_ndarray_fct = &GenericAxis<AxisValueType>::get_edges_ndarray;

        data_ = boost::shared_ptr< GenericAxisData<AxisValueType> >(new GenericAxisData<AxisValueType>());
        GenericAxisData<AxisValueType> & ddata = *static_cast<GenericAxisData<AxisValueType>*>(data_.get());
        intptr_t const nbins = edges.get_size();
        std::vector<intptr_t> shape(1, nbins);
        std::vector<intptr_t> front_capacity_vec(1, front_capacity);
        std::vector<intptr_t> back_capacity_vec(1, back_capacity);
        ddata.storage_ = boost::shared_ptr<detail::ndarray_storage>(
            new detail::ndarray_storage(shape, front_capacity_vec, back_capacity_vec, edges.get_dtype()));
        // Copy the data from the user provided edge array to the storage array.
        bp::object owner(bp::ptr(h));
        ddata.arr_ = ddata.storage_->ConstructNDArray(&owner);
        bn::ndarray & arr = *static_cast<bn::ndarray*>(&ddata.arr_);
        if(!bn::copy_into(arr, edges))
        {
            // TODO: Get the error string from the already set BP error.
            throw MemoryError("Could not copy edge array into internal storage!");
        }
        // Initialize a flat iterator over the axis edges.
        ddata.iter_ = bn::flat_iterator<AxisValueType>(arr);
    }

    static
    intptr_t
    get_bin_index(boost::shared_ptr<AxisData> axisdata, char * value_ptr)
    {
        GenericAxisData<AxisValueType> & data = *static_cast< GenericAxisData<AxisValueType> *>(axisdata.get());
        AxisValueType const & value = *reinterpret_cast<AxisValueType*>(value_ptr);

        // We know that edges is 1-dimensional by construction and the edges are
        // ordered ascedently. Also we know that the value type of the edges is
        // AxisValueType. So we can just iterate over the edges values and
        // compare the values.
        data.iter_.reset();
        for(intptr_t i=0; data.iter_ != data.iter_end_; ++data.iter_, ++i)
        {
            AxisValueType & lower_edge = *data.iter_;
            if(lower_edge > value)
            {
                return i-1;
            }
        }

        return -2;
    }

    static
    bn::ndarray
    get_edges_ndarray(boost::shared_ptr<AxisData> axisdata)
    {
        GenericAxisData<AxisValueType> & data = *static_cast< GenericAxisData<AxisValueType> *>(axisdata.get());
        bn::ndarray & edges_arr = *static_cast<bn::ndarray*>(&data.arr_);
        return edges_arr.copy();
    }
};

// Specialization for object axis types.
template <>
struct GenericAxis<bp::object>
  : Axis
{
    GenericAxis(::ndhist::ndhist * h, bn::ndarray const & edges, intptr_t front_capacity=0, intptr_t back_capacity=0)
    {
        // Set up the axis's function pointers.
        get_bin_index_fct     = &GenericAxis<bp::object>::get_bin_index;
        get_edges_ndarray_fct = &GenericAxis<bp::object>::get_edges_ndarray;

        data_ = boost::shared_ptr<GenericAxisData<bp::object> >(new GenericAxisData<bp::object>());
        GenericAxisData<bp::object> & ddata = *static_cast<GenericAxisData<bp::object>*>(data_.get());
        intptr_t const nbins = edges.get_size();
        std::vector<intptr_t> shape(1, nbins);
        std::vector<intptr_t> front_capacity_vec(1, front_capacity);
        std::vector<intptr_t> back_capacity_vec(1, back_capacity);
        ddata.storage_ = boost::shared_ptr<detail::ndarray_storage>(
            new detail::ndarray_storage(shape, front_capacity_vec, back_capacity_vec, edges.get_dtype()));
        // Copy the data from the user provided edge array to the storage array.
        bp::object owner(bp::ptr(h));
        ddata.arr_ = ddata.storage_->ConstructNDArray(&owner);
        bn::ndarray & arr = *static_cast<bn::ndarray*>(&ddata.arr_);
        if(!bn::copy_into(arr, edges))
        {
            // TODO: Get the error string from the already set BP error.
            throw MemoryError(
               "Could not copy edge array into internal storage!");
        }
        // Initialize a flat iterator over the axis edges.
        ddata.iter_ = bn::flat_iterator<bp::object>(arr);
    }

    static
    intptr_t
    get_bin_index(boost::shared_ptr<AxisData> axisdata, char * obj_ptr)
    {
        GenericAxisData<bp::object> & data = *static_cast<GenericAxisData<bp::object> *>(axisdata.get());
        uintptr_t * obj_ptr_data = reinterpret_cast<uintptr_t*>(obj_ptr);
        bp::object value(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*obj_ptr_data)));

        // We know that edges is 1-dimensional by construction and the edges are
        // ordered ascedently. Also we know that the value type of the edges is
        // AxisValueType. So we can just iterate over the edges values and
        // compare the values.
        data.iter_.reset();
        for(intptr_t i=0; data.iter_ != data.iter_end_; ++data.iter_, ++i)
        {
            bp::object lower_edge = *data.iter_;
            PyTypeObject* lower_edge_type = (PyTypeObject*)PyObject_Type(lower_edge.ptr());
            if(! PyObject_TypeCheck(value.ptr(), lower_edge_type))
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
                std::cout << "index = " << i-1 << std::endl;
                return i-1;
            }
        }

        std::cout << "index = " << -2 << std::endl;
        return -2;
    }

    static
    bn::ndarray
    get_edges_ndarray(boost::shared_ptr<AxisData> axisdata)
    {
        GenericAxisData<bp::object> & data = *static_cast< GenericAxisData<bp::object> *>(axisdata.get());
        bn::ndarray & edges_arr = *static_cast<bn::ndarray*>(&data.arr_);
        return edges_arr.copy();
    }
};



}//namespace detail
}//namespace ndhist

#endif // !NDHIST_DETAIL_GENERIC_AXIS_HPP_INCLUDED
