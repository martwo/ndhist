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
#include <sstream>
#include <boost/python/type_id.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/ndhist.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

ndhist::
ndhist(
    bn::ndarray const & shape
  , bp::list const & edges
  , bn::dtype const & dt
)
  : bc_(ConstructBinContentStorage(shape, dt))
{
    const size_t nd = shape.get_size();
    if(bp::len(edges) != nd)
    {
        std::stringstream ss;
        ss << "The size of the shape array (" << nd << ") and the length of "
           << "the edges list (" << bp::len(edges) << ") must be equal!";
        throw ValueError(ss.str());
    }

    // Construct the edges storages.
    bp::object self(bp::ptr(this));
    for(size_t i=0; i<nd; ++i)
    {
        const intptr_t n_bin_dim = bp::extract<intptr_t>(shape.get_item<bp::object>(i));
        bn::ndarray arr = bp::extract<bn::ndarray>(edges[i]);
        if(arr.get_nd() != 1)
        {
            std::stringstream ss;
            ss << "The dimension of the edges array for the " << i << "th "
               << "dimension of this histogram must be 1!";
            throw ValueError(ss.str());
        }
        if(arr.get_size() != n_bin_dim+1)
        {
            std::stringstream ss;
            ss << "The number of edges for the " << i << "th dimension of this "
               << "histogram must be " << n_bin_dim+1 << "!";
            throw ValueError(ss.str());
        }

        // Check the type of the edge values for the current axis.
        bn::dtype axis_dtype = arr.get_dtype();
        if(bn::dtype::equivalent(axis_dtype, bn::dtype::get_builtin<int64_t>()))
        {
            std::cout << "Found int64 equiv. edge type." << std::endl;
        }
        if(bn::dtype::equivalent(axis_dtype, bn::dtype::get_builtin<bp::object>()))
        {
            std::cout << "Found bp::object equiv. edge type." << std::endl;
        }

        std::vector<intptr_t> shape(1, n_bin_dim+1);
        std::vector<intptr_t> front_capacity(1, 0);
        std::vector<intptr_t> back_capacity(1, 0);
        boost::shared_ptr<detail::ndarray_storage> storage(
            new detail::ndarray_storage(shape, front_capacity, back_capacity, arr.get_dtype()));
        // Copy the data from the user provided edge array to the storage array.

        bn::ndarray storage_arr = storage->ConstructNDArray(&self);
        if(!bn::copy_into(storage_arr, arr))
        {
            throw MemoryError(
                "Could not copy edge array into internal storage!");
        }
        edges_storage_.push_back(storage);
        edges_.push_back(storage_arr);
    }
}

boost::shared_ptr<detail::ndarray_storage>
ndhist::
ConstructBinContentStorage(bn::ndarray const & shape, bn::dtype const & dt)
{
    if(shape.get_nd() != 1)
    {
        throw ValueError(
            "The shape array must be 1-dimensional!");
    }

    std::vector<intptr_t> shape_vec = shape.as_vector<intptr_t>();

    // This function does not allow to specify extra front and back
    // capacity, so we set it to zero for all dimensions.
    std::vector<intptr_t> front_capacity(shape.get_size(), 0);
    std::vector<intptr_t> back_capacity(shape.get_size(), 0);

    return boost::shared_ptr<detail::ndarray_storage>(
        new detail::ndarray_storage(shape_vec, front_capacity, back_capacity, dt));
}

bn::ndarray
ndhist::
GetBinContentArray()
{
    return bc_->ConstructNDArray();
}

bn::ndarray
ndhist::
GetEdgesArray(int axis)
{
    // Count axis from the back if axis is negative.
    if(axis < 0) {
        axis += edges_.size();
    }

    if(axis < 0 || axis >= edges_.size())
    {
        std::stringstream ss;
        ss << "The axis parameter must be in the interval "
           << "[0, " << edges_.size()-1 << "] or "
           << "[-1, -"<< edges_.size() <<"]!";
        throw IndexError(ss.str());
    }

    return edges_[axis];
}


intptr_t get_axis_bin_index(bn::ndarray const & edges, bp::object const & value)
{
    // We know that edges is 1-dimensional by construction and the edges are
    // ordered ascedently.
    intptr_t const N = edges.get_size();
    for(intptr_t i=0; i<N; ++i)
    {
        bp::object lower_edge = edges.get_item<bp::object, intptr_t>(i);
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

void
ndhist::
Fill(std::vector<bp::object> ndvalue, bp::object weight)
{
    // TODO: Implement function to get the i-th dimension index of the bin
    //       content array given the i-th value of the ndvalue array.
    //
    //       Create an Axis struct holding different implementations for this
    //       function depending on the axis edge value type.
    //bp::object self(bp::ptr(this));

    //std::cout << "ndvalue = [";
    for(size_t i=0; i<ndvalue.size(); ++i)
    {
        // Construct the ndarray representation from the edges storage for the
        // i-th dimension and set the owner the Py_None object.
        //bn::ndarray edges = edges_[i]->ConstructNDArray(&self);
        intptr_t axis_idx = get_axis_bin_index(edges_[i], ndvalue[i]);

      //  std::cout << axis_idx <<",";
    }
    //std::cout << "]" << std::endl;
}

}//namespace ndhist
