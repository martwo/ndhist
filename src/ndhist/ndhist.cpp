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

#include <boost/python/refcount.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/indexed_iterator.hpp>
#include <boost/numpy/dstream.hpp>

#include <ndhist/ndhist.hpp>
#include <ndhist/detail/axis.hpp>
#include <ndhist/detail/generic_axis.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

namespace detail {

template <typename AxisValueType>
struct axis_traits
{
    static
    boost::shared_ptr<Axis>
    construct_axis(ndhist * self, bn::ndarray const & edges, intptr_t front_capacity=0, intptr_t back_capacity=0)
    {
        // Check if the edges have a constant bin width,
        // thus the axis is linear.
        bool has_const_bin_width = true;
        bn::flat_iterator<AxisValueType> edges_iter(const_cast<bn::ndarray &>(edges));
        bn::flat_iterator<AxisValueType> edges_iter_end;
        AxisValueType & prev_value = *edges_iter;
        ++edges_iter;
        bool is_first_dist = true;
        AxisValueType first_dist;
        for(; edges_iter != edges_iter_end; ++edges_iter)
        {
            AxisValueType & this_value = *edges_iter;
            AxisValueType this_dist = this_value - prev_value;
            if(is_first_dist)
            {
                prev_value = this_value;
                first_dist = this_dist;
                is_first_dist = false;
            }
            else
            {
                if(this_dist == first_dist)
                {
                    prev_value = this_value;
                }
                else
                {
                    has_const_bin_width = false;
                    break;
                }
            }
        }
        if(has_const_bin_width)
        {
            std::cout << "+++++++++++++ Detected const bin width of " << first_dist << std::endl;
        }

        return boost::shared_ptr<detail::GenericAxis<double> >(new detail::GenericAxis<double>(self, edges, front_capacity, back_capacity));
    }

};

template <typename BCValueType>
struct fill_traits
{
    static
    void
    fill(ndhist & self, bp::object const & ndvalue_obj, bp::object const & weight_obj)
    {
        bn::ndarray ndvalue_arr = bn::from_object(ndvalue_obj, 0, 2, bn::ndarray::ALIGNED);
        bn::ndarray weight_arr = bn::from_object(weight_obj, bn::dtype::get_builtin<BCValueType>(), 0, 1, bn::ndarray::ALIGNED);

        // Construct an iterator for the input arrays. We use the loop service
        // of BoostNumpy that determines the number of loop
        // dimensions automatically and provides generalized universal
        // functions.
        typedef bn::dstream::mapping::detail::core_shape<1>::shape<-1>
                core_shape_t0;
        typedef bn::dstream::array_definition< core_shape_t0, void>
                in_arr_def0;
        typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
                core_shape_t1;
        typedef bn::dstream::array_definition< core_shape_t1, BCValueType>
                in_arr_def1;
        typedef bn::dstream::detail::loop_service_arity<2>::loop_service<in_arr_def0, in_arr_def1>
                loop_service_t;
        bn::dstream::detail::input_array_service<in_arr_def0> in_arr_service0(ndvalue_arr);
        bn::dstream::detail::input_array_service<in_arr_def1> in_arr_service1(weight_arr);
        loop_service_t loop_service(in_arr_service0, in_arr_service1);

        bn::detail::iter_operand_flags_t in_arr_iter_op_flags0 = bn::detail::iter_operand::flags::READONLY::value;
        bn::detail::iter_operand_flags_t in_arr_iter_op_flags1 = bn::detail::iter_operand::flags::READONLY::value;

        bn::detail::iter_operand in_arr_iter_op0( in_arr_service0.get_arr(), in_arr_iter_op_flags0, in_arr_service0.get_arr_bcr_data() );
        bn::detail::iter_operand in_arr_iter_op1( in_arr_service1.get_arr(), in_arr_iter_op_flags1, in_arr_service1.get_arr_bcr_data() );

        bn::detail::iter_flags_t iter_flags =
            bn::detail::iter::flags::REFS_OK::value
          | bn::detail::iter::flags::EXTERNAL_LOOP::value;
        bn::order_t order = bn::KEEPORDER;
        bn::casting_t casting = bn::NO_CASTING;
        intptr_t buffersize = 0;

        bn::detail::iter iter(
              iter_flags
            , order
            , casting
            , loop_service.get_loop_nd()
            , loop_service.get_loop_shape_data()
            , buffersize
            , in_arr_iter_op0
            , in_arr_iter_op1
        );
        iter.init_full_iteration();

        // Create an indexed iterator for the bin content array.
        bn::ndarray & bc_arr = self.GetBCArray();
        bn::indexed_iterator<BCValueType> bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);

        // Do the iteration.
        int const nd = bc_arr.get_nd();
        std::vector<intptr_t> indices(nd);
        std::vector<intptr_t> ndvalue_dim_indices(1);
        std::vector<intptr_t> const ndvalue_strides = iter.get_operand(0).get_strides_vector();
        bn::dstream::wiring::detail::iter_data_ptr</*ND=*/1, 0> ndvalue_iter_data_ptr(iter, 0, ndvalue_dim_indices, ndvalue_strides);
        do {
            intptr_t size = iter.get_inner_loop_size();
            while(size--)
            {
                // Fill the scalar into the bin content array.
                // Get the coordinate of the current ndvalue.
                bool is_overflown = false;
                for(size_t i=0; i<nd; ++i)
                {
                    ndvalue_dim_indices[0] = i; // Point to the next ndvalue.
                    boost::shared_ptr<detail::Axis> & axis = self.GetAxes()[i];
                    intptr_t axis_idx = axis->get_bin_index_fct(axis->data_, ndvalue_iter_data_ptr());
                    if(axis_idx >= 0)
                    {
                        indices[i] = axis_idx;
                    }
                    else
                    {
                        // The current value is out of the axis bounds.
                        // Just ignore it for now.
                        // TODO: Introduce an under- and overflow bin for each
                        //       each axis. Or resize the axis.
                        is_overflown = true;
                        break;
                    }
                }

                // Increase the bin content if the bin exists.
                if(!is_overflown)
                {
                    // Get the weight value for this fill iteration.
                    BCValueType & weight = *reinterpret_cast<BCValueType*>(iter.get_data(1));
                    // Jump to the location of the requested bin content.
                    bc_iter.jump_to(indices);
                    // Get a reference to the bin content's value.
                    BCValueType & bc_value = *bc_iter;
                    bc_value += weight;
                }

                // Jump to the next fill iteration.
                iter.add_inner_loop_strides_to_data_ptrs();
            }
        } while(iter.next());
    }
};

template <>
struct fill_traits<bp::object>
{
    static
    void
    fill(ndhist & self, bp::object const & ndvalue_obj, bp::object const & weight_obj)
    {
        bn::ndarray ndvalue_arr = bn::from_object(ndvalue_obj, 0, 2, bn::ndarray::ALIGNED);
        bn::ndarray weight_arr = bn::from_object(weight_obj, bn::dtype::get_builtin<bp::object>(), 0, 1, bn::ndarray::ALIGNED);

        // Construct an iterator for the input arrays. We use the loop service
        // of BoostNumpy that determines the number of loop
        // dimensions automatically and provides generalized universal
        // functions.
        typedef bn::dstream::mapping::detail::core_shape<1>::shape<-1>
                core_shape_t0;
        typedef bn::dstream::array_definition< core_shape_t0, void>
                in_arr_def0;
        typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
                core_shape_t1;
        typedef bn::dstream::array_definition< core_shape_t1, bp::object>
                in_arr_def1;
        typedef bn::dstream::detail::loop_service_arity<2>::loop_service<in_arr_def0, in_arr_def1>
                loop_service_t;

        bn::dstream::detail::input_array_service<in_arr_def0> in_arr_service0(ndvalue_arr);
        bn::dstream::detail::input_array_service<in_arr_def1> in_arr_service1(weight_arr);
        loop_service_t loop_service(in_arr_service0, in_arr_service1);

        bn::detail::iter_operand_flags_t in_arr_iter_op_flags0 = bn::detail::iter_operand::flags::READONLY::value;
        bn::detail::iter_operand_flags_t in_arr_iter_op_flags1 = bn::detail::iter_operand::flags::READONLY::value;

        bn::detail::iter_operand in_arr_iter_op0( in_arr_service0.get_arr(), in_arr_iter_op_flags0, in_arr_service0.get_arr_bcr_data() );
        bn::detail::iter_operand in_arr_iter_op1( in_arr_service1.get_arr(), in_arr_iter_op_flags1, in_arr_service1.get_arr_bcr_data() );

        bn::detail::iter_flags_t iter_flags =
            bn::detail::iter::flags::EXTERNAL_LOOP::value
          | bn::detail::iter::flags::REFS_OK::value;
        bn::order_t order = bn::KEEPORDER;
        bn::casting_t casting = bn::NO_CASTING;
        intptr_t buffersize = 0;

        bn::detail::iter iter(
              iter_flags
            , order
            , casting
            , loop_service.get_loop_nd()
            , loop_service.get_loop_shape_data()
            , buffersize
            , in_arr_iter_op0
            , in_arr_iter_op1
        );
        iter.init_full_iteration();

        // Create an indexed iterator for the bin content array.
        bn::ndarray & bc_arr = self.GetBCArray();
        bn::indexed_iterator<bp::object> bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);

        // Do the iteration.
        int const nd = bc_arr.get_nd();
        std::vector<intptr_t> indices(nd);
        std::vector<intptr_t> ndvalue_dim_indices(1);
        std::vector<intptr_t> const ndvalue_strides = iter.get_operand(0).get_strides_vector();
        bn::dstream::wiring::detail::iter_data_ptr</*ND=*/1, 0> ndvalue_iter_data_ptr(iter, 0, ndvalue_dim_indices, ndvalue_strides);
        do {
            intptr_t size = iter.get_inner_loop_size();
            while(size--)
            {
                // Fill the scalar into the bin content array.
                // Get the coordinate of the current ndvalue.
                bool is_overflown = false;
                for(size_t i=0; i<nd; ++i)
                {
                    ndvalue_dim_indices[0] = i; // Point to the next ndvalue.
                    boost::shared_ptr<detail::Axis> & axis = self.GetAxes()[i];
                    intptr_t axis_idx = axis->get_bin_index_fct(axis->data_, ndvalue_iter_data_ptr());
                    if(axis_idx >= 0)
                    {
                        indices[i] = axis_idx;
                    }
                    else
                    {
                        // The current value is out of the axis bounds.
                        // Just ignore it for now.
                        // TODO: Introduce an under- and overflow bin for each
                        //       each axis. Or resize the axis.
                        is_overflown = true;
                        break;
                    }
                }

                // Increase the bin content if the bin exists.
                if(!is_overflown)
                {
                    // Get the weight value for this fill iteration.
                    uintptr_t * weight_ptr = reinterpret_cast<uintptr_t*>(iter.get_data(1));
                    bp::object weight(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*weight_ptr)));

                    // Jump to the location of the requested bin content.
                    bc_iter.jump_to(indices);

                    bp::object bc_value = *bc_iter;
                    // Use the bp::object operator+= implementation, which will
                    // call the appropriate Python function of the object.
                    bc_value += weight;
                }

                // Jump to the next fill iteration.
                iter.add_inner_loop_strides_to_data_ptrs();
            }
        } while(iter.next());
    }
};

}// namespace detail

ndhist::
ndhist(
    bn::ndarray const & shape
  , bp::list const & edges
  , bn::dtype const & dt
  , bp::object const & bc_class
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

    // Initialize the bin content array with objects using their default
    // constructor when the bin content array is an object array.
    if(bn::dtype::equivalent(bc_dtype, bn::dtype::get_builtin<bp::object>()))
    {
        bn::flat_iterator<bp::object> bc_iter(GetBCArray());
        bn::flat_iterator<bp::object> bc_iter_end;
        for(; bc_iter != bc_iter_end; ++bc_iter)
        {
            uintptr_t * obj_ptr_ptr = bc_iter.get_object_ptr_ptr();
            bp::object obj = bc_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }
    }

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
            axes_.push_back(detail::axis_traits<double>::construct_axis(this, arr, 0, 0));
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

bn::ndarray
ndhist::
get_edges_ndarray(intptr_t axis) const
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

    return axes_[axis]->get_edges_ndarray_fct(axes_[axis]->data_);
}

void
ndhist::
Fill(bp::object const & ndvalue_obj, bp::object const & weight_obj)
{
    fill_fct_(*this, ndvalue_obj, weight_obj);
}

}//namespace ndhist
