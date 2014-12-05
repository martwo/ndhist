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
        axes_.push_back(boost::shared_ptr<detail::GenericAxis>(new detail::GenericAxis(this, arr, 0, 0)));
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
/*
bn::ndarray
ndhist::
GetEdgesArray(int axis)
{
    // Count axis from the back if axis is negative.
    if(axis < 0) {
        axis += axes_.size();
    }

    if(axis < 0 || axis >= axes_.size())
    {
        std::stringstream ss;
        ss << "The axis parameter must be in the interval "
           << "[0, " << axes_.size()-1 << "] or "
           << "[-1, -"<< axes_.size() <<"]!";
        throw IndexError(ss.str());
    }

    return axes_[axis]->get_edges_ndarray_fct(axes_[axis]->data);
}
*/



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

    std::cout << "ndvalue = [";
    for(size_t i=0; i<ndvalue.size(); ++i)
    {
        // Construct the ndarray representation from the edges storage for the
        // i-th dimension and set the owner the Py_None object.
        //bn::ndarray edges = edges_[i]->ConstructNDArray(&self);
        intptr_t axis_idx = axes_[i]->get_bin_index_fct(axes_[i]->data_, ndvalue[i]);

        std::cout << axis_idx <<",";
    }
    std::cout << "]" << std::endl;
}

}//namespace ndhist
