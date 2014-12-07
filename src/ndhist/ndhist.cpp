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

#include <boost/numpy/ndarray.hpp>

#include <ndhist/detail/generic_axis.hpp>
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
    // Create a ndarray for the bin content.
    bp::object self(bp::ptr(this));
    bc_arr_ = bc_->ConstructNDArray(&self);

    const size_t nd = shape.get_size();
    if(bp::len(edges) != nd)
    {
        std::stringstream ss;
        ss << "The size of the shape array (" << nd << ") and the length of "
           << "the edges list (" << bp::len(edges) << ") must be equal!";
        throw ValueError(ss.str());
    }

    // Construct the axes of the histogram.
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
            ss << "The number of edges for the " << i+1 << "th dimension of this "
               << "histogram must be " << n_bin_dim+1 << "!";
            throw ValueError(ss.str());
        }

        // Check the type of the edge values for the current axis.
        bn::dtype axis_dtype = arr.get_dtype();
        if(bn::dtype::equivalent(axis_dtype, bn::dtype::get_builtin<double>()))
        {
            std::cout << "Found double equiv. edge type." << std::endl;
            axes_.push_back(boost::shared_ptr<detail::GenericAxis<double> >(new detail::GenericAxis<double>(this, arr, 0, 0)));
        }

        else if(bn::dtype::equivalent(axis_dtype, bn::dtype::get_builtin<bp::object>()))
        {
            std::cout << "Found bp::object equiv. edge type." << std::endl;
            axes_.push_back(boost::shared_ptr<detail::GenericAxis<bp::object> >(new detail::GenericAxis<bp::object>(this, arr, 0, 0)));

        }
        else
        {
            std::stringstream ss;
            ss << "The data type of the edges of axis "<< i+1<< " is not "
               << "supported.";
            throw TypeError(ss.str());
        }
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
    bn::ndarray & bc_arr = *static_cast<bn::ndarray*>(&bc_arr_);
    int const nd = bc_arr.get_nd();

    //std::cout << "ndvalue = [";
    bp::list indices;
    std::vector<intptr_t> indices_vec(nd);

    for(size_t i=0; i<ndvalue.size(); ++i)
    {
        intptr_t axis_idx = axes_[i]->get_bin_index_fct(axes_[i]->data_, ndvalue[i]);
        if(axis_idx >= 0) {
            std::cout << "Add "<< axis_idx<<" for axis = "<<i<<std::endl;
            indices.append(axis_idx);
            indices_vec[i] = axis_idx;
        }
        else {
            // The current value is out of the axis bounds. Just ignore it
            // for the moment.
            return;
        }
        //std::cout << axis_idx <<",";
    }


    bn::detail::iter_flags_t iter_flags =
        bn::detail::iter::flags::MULTI_INDEX::value
      | bn::detail::iter::flags::DONT_NEGATE_STRIDES::value;

    intptr_t itershape[nd];
    int arr_op_bcr[nd];
    for(int i=0; i<nd; ++i)
    {
        itershape[i] = -1;
        arr_op_bcr[i] = i;
    }
    bn::detail::iter_operand_flags_t arr_op_flags = bn::detail::iter_operand::flags::READONLY::value;
    bn::detail::iter_operand arr_op(bc_arr, arr_op_flags, arr_op_bcr);
    bn::detail::iter iter(
        iter_flags
      , bn::KEEPORDER
      , bn::NO_CASTING
      , nd           // n_iter_axes
      , itershape
      , 0           // buffersize
      , arr_op
    );
    iter.init_full_iteration();
    iter.go_to(indices_vec);

    int64_t & value = *reinterpret_cast<int64_t*>(iter.get_data(0));
    int64_t w = bp::extract<int64_t>(weight);
    value += w;
    /*

    bp::object bc = bc_arr[indices].scalarize();
    bool s = bn::is_any_scalar(bc);
    bc += weight;
    //int64_t c = bp::extract<int64_t>(bc);
    std::cout << "s = " << s << std::endl;
    //std::cout << "c = " << c << std::endl;
    //bc += bp::extract<int64_t>(weight);
    //std::cout << "]" << std::endl;
    */
}

}//namespace ndhist
