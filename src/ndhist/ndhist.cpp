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

namespace detail {

template <typename ValueType>
struct fill_traits
{
    static
    void
    fill(ndhist & self, std::vector<bp::object> const & ndvalue, bp::object const & weight)
    {
        bn::ndarray & bc_arr = self.GetBCArray();
        int const nd = bc_arr.get_nd();
        assert(nd == ndvalue.size());
        std::vector<intptr_t> indices(nd);
        intptr_t itershape[nd];
        int arr_op_bcr[nd];
        for(size_t i=0; i<nd; ++i)
        {
            boost::shared_ptr<detail::Axis> & axis = self.GetAxes()[i];
            intptr_t axis_idx = axis->get_bin_index_fct(axis->data_, ndvalue[i]);
            if(axis_idx >= 0)
            {
                indices[i] = axis_idx;
            }
            else
            {
                // The current value is out of the axis bounds. Just ignore it
                // for the moment.
                return;
            }
            itershape[i] = -1;
            arr_op_bcr[i] = i;
        }

        bn::detail::iter_flags_t iter_flags =
            bn::detail::iter::flags::MULTI_INDEX::value
          | bn::detail::iter::flags::DONT_NEGATE_STRIDES::value;

        bn::detail::iter_operand_flags_t arr_op_flags = bn::detail::iter_operand::flags::READONLY::value;
        bn::detail::iter_operand arr_op(bc_arr, arr_op_flags, arr_op_bcr);
        bn::detail::iter iter(
            iter_flags
          , bn::KEEPORDER
          , bn::NO_CASTING
          , nd           // n_iter_axes
          , itershape
          , 0            // buffersize
          , arr_op
        );
        iter.init_full_iteration();
        iter.go_to(indices);

        ValueType & value = *reinterpret_cast<ValueType*>(iter.get_data(0));
        ValueType w = bp::extract<ValueType>(weight);
        value += w;
    }
};

template <>
struct fill_traits<bp::object>
{
    static
    void
    fill(ndhist & self, std::vector<bp::object> const & ndvalue, bp::object const & weight)
    {
        size_t const coordnd = ndvalue.size();
        bp::list indices;
        for(size_t i=0; i<coordnd; ++i)
        {
            boost::shared_ptr<detail::Axis> & axis = self.GetAxes()[i];
            intptr_t axis_idx = axis->get_bin_index_fct(axis->data_, ndvalue[i]);
            if(axis_idx >= 0) {
                std::cout << "Add "<< axis_idx<<" for axis = "<<i<<std::endl;
                indices.append(axis_idx);
            }
            else {
                // The current value is out of the axis bounds. Just ignore it
                // for the moment.
                return;
            }
        }

        bn::ndarray & bc_arr = self.GetBCArray();
        bp::object bc = bc_arr[indices].scalarize();
        bc += weight;
    }
};

}// namespace detail

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

    #define NDHIST_BC_DATA_TYPE_SUPPORT(BCDTYPE)                               \
    if(bn::dtype::equivalent(bc_dtype, bn::dtype::get_builtin<BCDTYPE>()))     \
    {                                                                          \
        if(bc_dtype_supported) {                                               \
            std::stringstream ss;                                              \
            ss << "The bin content data type is supported by more than one "   \
               << "possible C++ data type! This is an internal error!";        \
            throw TypeError(ss.str());                                         \
        }                                                                      \
        std::cout << "Found " << BOOST_PP_STRINGIZE(BCDTYPE) << " equiv. "     \
                  << "bc data type." << std::endl;                             \
        fill_fct_ = &detail::fill_traits<BCDTYPE>::fill;                       \
        bc_dtype_supported = true;                                             \
    }

    // Set the fill function based on the bin content data type.
    bn::dtype bc_dtype = GetBCArray().get_dtype();
    bool bc_dtype_supported = false;
    NDHIST_BC_DATA_TYPE_SUPPORT(bool)
    NDHIST_BC_DATA_TYPE_SUPPORT(int16_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(uint16_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(int32_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(uint32_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(int64_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(uint64_t)
    NDHIST_BC_DATA_TYPE_SUPPORT(float)
    NDHIST_BC_DATA_TYPE_SUPPORT(double)
    NDHIST_BC_DATA_TYPE_SUPPORT(bp::object)
    if(!bc_dtype_supported)
    {
        std::stringstream ss;
        ss << "The data type of the bin content array is not supported.";
        throw TypeError(ss.str());
    }
    #undef NDHIST_BC_DATA_TYPE_SUPPORT

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
    fill_fct_(*this, ndvalue, weight);
}

}//namespace ndhist
