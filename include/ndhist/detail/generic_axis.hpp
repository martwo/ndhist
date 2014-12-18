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

#include <algorithm>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

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

template <class Derived, class DerivedData>
struct GenericAxisBase
  : Axis
{
    GenericAxisBase(
        ::ndhist::ndhist * h
      , bn::ndarray const & edges
      , intptr_t front_capacity=0
      , intptr_t back_capacity=0
    )
      : Axis(edges.get_dtype())
    {
        // Set up the axis's function pointers.
        get_bin_index_fct     = &Derived::get_bin_index;
        get_edges_ndarray_fct = &Derived::get_edges_ndarray;

        data_ = boost::shared_ptr< DerivedData >(new DerivedData());
        DerivedData & ddata = *static_cast<DerivedData*>(data_.get());
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
        ddata.iter_ = bn::flat_iterator<typename Derived::axis_value_type>(arr);
        ddata.iter_end_ = bn::flat_iterator<typename Derived::axis_value_type>(ddata.iter_.end());
    }

    static
    bn::ndarray
    get_edges_ndarray(boost::shared_ptr<AxisData> axisdata)
    {
        DerivedData & data = *static_cast<DerivedData*>(axisdata.get());
        bn::ndarray & edges_arr = *static_cast<bn::ndarray*>(&data.arr_);
        return edges_arr.copy();
    }
};

template <typename AxisValueType>
struct GenericAxis
  : GenericAxisBase< GenericAxis<AxisValueType>, GenericAxisData<AxisValueType> >
{
    typedef GenericAxisBase< GenericAxis<AxisValueType>, GenericAxisData<AxisValueType> >
            base_t;
    typedef AxisValueType axis_value_type;

    GenericAxis(::ndhist::ndhist * h, bn::ndarray const & edges, intptr_t front_capacity=0, intptr_t back_capacity=0)
      : base_t(h, edges, front_capacity, back_capacity)
    {}

    static
    intptr_t
    get_bin_index(boost::shared_ptr<AxisData> axisdata, char * value_ptr)
    {
        GenericAxisData<AxisValueType> & data = *static_cast< GenericAxisData<AxisValueType> *>(axisdata.get());
        AxisValueType const & value = *reinterpret_cast<AxisValueType*>(value_ptr);

        // We know that edges is 1-dimensional by construction and the edges are
        // ordered ascedently. Also we know that the value type of the edges is
        // AxisValueType. So we can use the std::upper_bound binary search for
        // getting the upper edge for the given value.
        data.iter_.reset();
        bn::flat_iterator<AxisValueType> ub = std::upper_bound(data.iter_, data.iter_end_, value);
        if(ub == data.iter_end_)
        {
            // Overlow.
            return -2;
        }
        intptr_t const idx = ub.get_iter_index();
        if(idx == 0)
        {
            // Underflow. ub points to the first element.
            return -1;
        }
        return idx - 1;
    }
};

// Specialization for object axis types.
template <>
struct GenericAxis<bp::object>
  : GenericAxisBase< GenericAxis<bp::object>, GenericAxisData<bp::object> >
{
    typedef GenericAxisBase< GenericAxis<bp::object>, GenericAxisData<bp::object> >
            base_t;
    typedef bp::object axis_value_type;

    GenericAxis(::ndhist::ndhist * h, bn::ndarray const & edges, intptr_t front_capacity=0, intptr_t back_capacity=0)
      : base_t(h, edges, front_capacity, back_capacity)
    {}

    static
    bool
    edge_value_compare(bp::object const & value, bp::object const & edge)
    {
        bp::object edge_type(bp::handle<>(bp::borrowed(bp::downcast<PyTypeObject>(PyObject_Type(edge.ptr())))));
        if(! PyObject_TypeCheck(value.ptr(), (PyTypeObject*)edge_type.ptr()))
        {
            std::stringstream ss;
            ss << "The type of the value for an axis must be the same as the "
               << "type of the edges objects of that axis! "
               << "Otherwise comparison operators might be ill-defined.";
            throw TypeError(ss.str());
        }
        return (value < edge);
    }

    static
    intptr_t
    get_bin_index(boost::shared_ptr<AxisData> axisdata, char * obj_ptr)
    {
        std::cout << "GenericAxis<bp::object>::get_bin_index" << std::endl;
        GenericAxisData<bp::object> & data = *static_cast<GenericAxisData<bp::object> *>(axisdata.get());
        uintptr_t * obj_ptr_data = reinterpret_cast<uintptr_t*>(obj_ptr);
        bp::object value(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*obj_ptr_data)));

        // We know that edges is 1-dimensional by construction and the edges are
        // ordered ascedently. Also we know that the value type of the edges is
        // AxisValueType. So we can just iterate over the edges values and
        // compare the values.
        data.iter_.reset();
        bn::flat_iterator<bp::object> ub = std::upper_bound(data.iter_, data.iter_end_, value, &edge_value_compare);
        if(ub == data.iter_end_)
        {
            // Overlow.
            return -2;
        }
        intptr_t const idx = ub.get_iter_index();
        if(idx == 0)
        {
            // Underflow. ub points to the first element.
            return -1;
        }
        return idx - 1;
    }
};

}//namespace detail
}//namespace ndhist

#endif // !NDHIST_DETAIL_GENERIC_AXIS_HPP_INCLUDED
