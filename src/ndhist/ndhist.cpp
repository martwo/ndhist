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

#include <boost/type_traits/is_same.hpp>

#include <boost/python/refcount.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>
#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/indexed_iterator.hpp>
#include <boost/numpy/dstream.hpp>

#include <ndhist/ndhist.hpp>
#include <ndhist/detail/axis.hpp>
#include <ndhist/detail/limits.hpp>
#include <ndhist/detail/generic_axis.hpp>
#include <ndhist/detail/constant_bin_width_axis.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

namespace detail {

template <typename AxisValueType>
struct axis_traits
{
    static bool
    has_constant_bin_width(bn::ndarray const & edges)
    {
        bn::flat_iterator<AxisValueType> edges_iter(edges);
        bn::flat_iterator<AxisValueType> const edges_iter_end(edges_iter.end());
        AxisValueType prev_value = *edges_iter;
        ++edges_iter;
        bool is_first_dist = true;
        AxisValueType first_dist;
        for(; edges_iter != edges_iter_end; ++edges_iter)
        {
            AxisValueType this_value = *edges_iter;
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
                    return false;
                }
            }
        }

        return true;
    }

    static
    boost::shared_ptr<Axis>
    construct_axis(ndhist * self, bn::ndarray const & edges, intptr_t autoscale_fcap=0, intptr_t autoscale_bcap=0)
    {
        // Check if the edges have a constant bin width,
        // thus the axis is linear.
        if(has_constant_bin_width(edges))
        {
            std::cout << "+++++++++++++ Detected const bin width of "  << std::endl;
            return boost::shared_ptr<detail::ConstantBinWidthAxis<AxisValueType> >(new detail::ConstantBinWidthAxis<AxisValueType>(edges, autoscale_fcap, autoscale_bcap));
        }

        return boost::shared_ptr< detail::GenericAxis<AxisValueType> >(new detail::GenericAxis<AxisValueType>(self, edges));
    }

};

template <>
struct axis_traits<bp::object>
{
    static
    boost::shared_ptr<Axis>
    construct_axis(ndhist * self, bn::ndarray const & edges, intptr_t autoscale_fcap=0, intptr_t autoscale_bcap=0)
    {
        // In case we have an object value typed axis, we use the
        // GenericAxis, because it requires only the < comparison
        // operator.
        return boost::shared_ptr< detail::GenericAxis<bp::object> >(new detail::GenericAxis<bp::object>(self, edges));
    }
};

// The implementation to increment the bin content by a weight based on the type
// of the bin content value.
template <typename BCValueType>
void increment_bin_content(
    bn::detail::iter & in_iter
  , int weight_arr_op_idx
  , bn::indexed_iterator<BCValueType> & bc_iter
  , std::vector<intptr_t> const & indices
)
{
    // Get the weight value from the input iterator.
    BCValueType & weight = *reinterpret_cast<BCValueType*>(in_iter.get_data(weight_arr_op_idx));
    // Jump to the location of the requested bin content.
    bc_iter.jump_to(indices);
    // Get a reference to the bin content's value.
    BCValueType & bc_value = *bc_iter;
    bc_value += weight;
}

template <>
void increment_bin_content<bp::object>(
    bn::detail::iter & in_iter
  , int weight_arr_op_idx
  , bn::indexed_iterator<bp::object> & bc_iter
  , std::vector<intptr_t> const & indices
)
{
    // Get the weight value from the input iterator.
    uintptr_t * weight_ptr = reinterpret_cast<uintptr_t*>(in_iter.get_data(weight_arr_op_idx));
    bp::object weight(bp::detail::borrowed_reference(reinterpret_cast<PyObject*>(*weight_ptr)));

    // Jump to the location of the requested bin content.
    bc_iter.jump_to(indices);

    bp::object bc_value = *bc_iter;
    // Use the bp::object operator+= implementation, which will
    // call the appropriate Python function of the object.
    bc_value += weight;
}

struct generic_nd_traits
{
    template <typename BCValueType>
    struct fill_traits
    {
        static
        void
        fill(ndhist & self, bp::object const & ndvalues_obj, bp::object const & weight_obj)
        {
            // The ndvalues_obj object is supposed to be a structured ndarray.
            // But in case of 1-dimensional histogram we accept also a
            // simple array, which can also be a normal Python list.

            int const nd = self.get_nd();
            bp::object ndvalues_arr_obj;
            try
            {
                ndvalues_arr_obj = bn::from_object(ndvalues_obj, self.get_ndvalues_dtype(), 0, 0, bn::ndarray::ALIGNED);
            }
            catch (const bp::error_already_set &)
            {
                if(nd == 1)
                {
                    ndvalues_arr_obj = bn::from_object(ndvalues_obj, self.get_axes()[0]->get_dtype(), 0, 0, bn::ndarray::ALIGNED);
                }
                else
                {
                    std::stringstream ss;
                    ss << "The ndvalues parameter must either be a tuple of "
                       << nd << " one-dimensional arrays or one structured "
                       << "ndarray!";
                    throw TypeError(ss.str());
                }
            }
            bn::ndarray ndvalue_arr = *static_cast<bn::ndarray*>(&ndvalues_arr_obj);
            bn::ndarray weight_arr = bn::from_object(weight_obj, bn::dtype::get_builtin<BCValueType>(), 0, 0, bn::ndarray::ALIGNED);

            // Construct an iterator for the input arrays. We use the loop service
            // of BoostNumpy that determines the number of loop
            // dimensions automatically and provides generalized universal
            // functions.
            typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
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
                bn::detail::iter::flags::REFS_OK::value // This is needed for the
                                                        // weight, which can be bp::object.
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
            // Get the byte offsets of the fields and check if the number of fields
            // match the dimensionality of the histogram.
            std::vector<intptr_t> ndvalue_byte_offsets = ndvalue_arr.get_dtype().get_fields_byte_offsets();

            if(ndvalue_byte_offsets.size() == 0)
            {
                if(nd == 1)
                {
                    ndvalue_byte_offsets.push_back(0);
                }
                else
                {
                    std::stringstream ss;
                    ss << "The dimensionality of the histogram is " << nd
                    << ", i.e. greater than 1, so the value ndarray must be a "
                    << "structured ndarray!";
                    throw ValueError(ss.str());
                }
            }
            else if(ndvalue_byte_offsets.size() != nd)
            {
                std::stringstream ss;
                ss << "The value ndarray must contain " << nd << " fields, one for "
                   << "each dimension! Right now it has "
                   << ndvalue_byte_offsets.size() << " fields!";
                throw ValueError(ss.str());
            }

            // Do the iteration.
            std::vector<intptr_t> indices(nd);
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Fill the scalar into the bin content array.
                    // Get the coordinate of the current ndvalue.
                    bool is_overflown = false;
                    for(size_t i=0; i<nd; ++i)
                    {
                        std::cout << "Get bin idx of axis " << i << " of " << nd << std::endl;
                        boost::shared_ptr<detail::Axis> & axis = self.get_axes()[i];
                        char * ndvalue_ptr = iter.get_data(0) + ndvalue_byte_offsets[i];
                        axis::out_of_range_t oor;
                        intptr_t axis_idx = axis->get_bin_index_fct(axis->data_, ndvalue_ptr, &oor);
                        if(oor == axis::OOR_NONE)
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
                        increment_bin_content<BCValueType>(iter, 1, bc_iter, indices);
                    }

                    // Jump to the next fill iteration.
                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    }; // struct fill_traits
}; // struct generic_nd_traits

template <int nd>
struct nd_traits;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND, <ndhist/ndhist.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail

ndhist::
ndhist(
    bp::tuple const & axes
  , bn::dtype const & dt
  , bp::object const & bc_class
)
  : ndvalues_dt_(bn::dtype::new_builtin<void>())
{
    int const nd = bp::len(axes);
    std::vector<intptr_t> shape(nd);
    std::vector<intptr_t> front_capacity(nd);
    std::vector<intptr_t> back_capacity(nd);

    // Construct the axes of the histogram.
    for(size_t i=0; i<nd; ++i)
    {
        // Each axes element can be a single ndarray or a tuple of the form
        // (edges_ndarry[, axis_name[, front_capacity, back_capacity]])
        bp::object axis = axes[i];
        bp::object edges_arr_obj;
        bp::object axis_name_obj;
        bp::object fcap_obj;
        bp::object bcap_obj;
        if(PyTuple_Check(axis.ptr()))
        {
            size_t const tuple_len = bp::len(axis);
            if(tuple_len == 0)
            {
                std::stringstream ss;
                ss << "The "<<i<<"th axis tuple is empty!";
                throw ValueError(ss.str());
            }
            else if(tuple_len == 1)
            {
                edges_arr_obj = axis[0];
                std::stringstream axis_name;
                axis_name << "a" << i;
                axis_name_obj = bp::str(axis_name.str());
                fcap_obj = bp::object(0);
                bcap_obj = bp::object(0);
            }
            else if(tuple_len == 2)
            {
                edges_arr_obj = axis[0];
                axis_name_obj = axis[1];
                fcap_obj = bp::object(0);
                bcap_obj = bp::object(0);
            }
            else if(tuple_len == 4)
            {
                edges_arr_obj = axis[0];
                axis_name_obj = axis[1];
                fcap_obj      = axis[2];
                bcap_obj      = axis[3];
            }
            else
            {
                std::stringstream ss;
                ss << "The "<<i<<"th axis tuple must have a length of "
                   << "either 1, 2, or 4!";
                throw ValueError(ss.str());
            }
        }
        else
        {
            // Only the edges array is given.
            edges_arr_obj = axis;
            std::stringstream axis_name;
            axis_name << "a" << i;
            axis_name_obj = bp::str(axis_name.str());
            fcap_obj = bp::object(0);
            bcap_obj = bp::object(0);
        }

        bn::ndarray edges_arr = bn::from_object(edges_arr_obj, 0, 1, bn::ndarray::ALIGNED);
        if(edges_arr.get_nd() != 1)
        {
            std::stringstream ss;
            ss << "The dimension of the edges array for the " << i+1 << "th "
               << "dimension of this histogram must be 1!";
            throw ValueError(ss.str());
        }
        const intptr_t n_bin_dim = edges_arr.get_size();

        intptr_t const autoscale_fcap = bp::extract<intptr_t>(fcap_obj);
        intptr_t const autoscale_bcap = bp::extract<intptr_t>(bcap_obj);

        // Check the type of the edge values for the current axis.
        bool axis_dtype_supported = false;
        bn::dtype axis_dtype = edges_arr.get_dtype();
        #define NDHIST_AXIS_DATA_TYPE_SUPPORT(AXISDTYPE)                            \
            if(bn::dtype::equivalent(axis_dtype, bn::dtype::get_builtin<AXISDTYPE>()))\
            {                                                                       \
                if(axis_dtype_supported) {                                          \
                    std::stringstream ss;                                           \
                    ss << "The bin content data type is supported by more than one "\
                       << "possible C++ data type! This is an internal error!";     \
                    throw TypeError(ss.str());                                      \
                }                                                                   \
                std::cout << "Found " << BOOST_PP_STRINGIZE(AXISDTYPE) << " equiv. "\
                          << "axis data type." << std::endl;                        \
                axes_.push_back(detail::axis_traits<AXISDTYPE>::construct_axis(this, edges_arr, autoscale_fcap, autoscale_bcap));\
                axis_dtype_supported = true;                                        \
            }
        NDHIST_AXIS_DATA_TYPE_SUPPORT(int16_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(uint16_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(int32_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(uint32_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(int64_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(uint64_t)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(float)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(double)
        NDHIST_AXIS_DATA_TYPE_SUPPORT(bp::object)
        if(!axis_dtype_supported)
        {
            std::stringstream ss;
            ss << "The data type of the edges of axis "<< i << " is not "
               << "supported.";
            throw TypeError(ss.str());
        }
        #undef NDHIST_AXIS_DATA_TYPE_SUPPORT

        // Add the axis field to the ndvalues dtype object.
        std::string field_name = bp::extract<std::string>(axis_name_obj);
        ndvalues_dt_.add_field(field_name, axes_[i]->get_dtype());

        // Add the shape information for this axis.
        shape[i] = n_bin_dim;

        // Set the extra front and back capacity for this axis if the axis has
        // an autoscale.
        if(axes_[i]->has_autoscale())
        {
            front_capacity[i] = autoscale_fcap;
            back_capacity[i]  = autoscale_bcap;
        }
        else
        {
            front_capacity[i] = 0;
            back_capacity[i]  = 0;
        }
    }

    // Create a ndarray for the bin content.
    bc_ = boost::shared_ptr<detail::ndarray_storage>(new detail::ndarray_storage(shape, front_capacity, back_capacity, dt));
    bp::object self(bp::ptr(this));
    bc_arr_ = bc_->ConstructNDArray(&self);

    // Set the fill function based on the bin content data type.
    bn::dtype bc_dtype = GetBCArray().get_dtype();
    bool bc_dtype_supported = false;
    #define BOOST_PP_ITERATION_PARAMS_1                                        \
        (4, (1, NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND, <ndhist/ndhist.hpp>, 2))
    #include BOOST_PP_ITERATE()
    else
    {
        // nd is greater than NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND.
        #define NDHIST_BC_DATA_TYPE_SUPPORT(BCDTYPE)                           \
            if(bn::dtype::equivalent(bc_dtype, bn::dtype::get_builtin<BCDTYPE>()))  \
            {                                                                       \
                if(bc_dtype_supported) {                                            \
                    std::stringstream ss;                                           \
                    ss << "The bin content data type is supported by more than one "\
                    << "possible C++ data type! This is an internal error!";        \
                    throw TypeError(ss.str());                                      \
                }                                                                   \
                std::cout << "Found " << BOOST_PP_STRINGIZE(BCDTYPE) << " equiv. "  \
                          << "bc data type." << std::endl;                          \
                fill_fct_ = &detail::generic_nd_traits::fill_traits<BCDTYPE>::fill; \
                bc_dtype_supported = true;                                          \
            }
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
        #undef NDHIST_BC_DATA_TYPE_SUPPORT
    }
    if(!bc_dtype_supported)
    {
        std::stringstream ss;
        ss << "The data type of the bin content array is not supported.";
        throw TypeError(ss.str());
    }

    // Initialize the bin content array with objects using their default
    // constructor when the bin content array is an object array.
    if(bn::dtype::equivalent(bc_dtype, bn::dtype::get_builtin<bp::object>()))
    {
        bn::flat_iterator<bp::object> bc_iter(GetBCArray());
        bn::flat_iterator<bp::object> bc_iter_end(bc_iter.end());
        for(; bc_iter != bc_iter_end; ++bc_iter)
        {
            bool isnone = (bc_class == bp::object());
            uintptr_t * obj_ptr_ptr = bc_iter.get_object_ptr_ptr();
            bp::object obj = bc_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }

        bc_one_ = bc_class(1);
    }
    else
    {
        bp::object one(1);
        bc_one_ = bn::from_object(one, GetBCArray().get_dtype(), 0, 1, bn::ndarray::ALIGNED).scalarize();
    }
}

bn::ndarray
ndhist::
py_construct_bin_content_ndarray()
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
fill(bp::object const & ndvalue_obj, bp::object weight_obj)
{
    // In case None is given as weight, we will use one.
    if(weight_obj == bp::object())
    {
        weight_obj = this->get_one();
    }
    fill_fct_(*this, ndvalue_obj, weight_obj);
}

void
ndhist::
extend_bin_content_array_axis(intptr_t axis, intptr_t n_extra_bins)
{
    if(n_extra_bins == 0)
        return;
    std::vector<intptr_t> old_shape = GetBCArray().get_shape_vector();

    bc_->extend_axis(axis, n_extra_bins);
    // Recreate the bin content ndarray.
    bp::object self(bp::ptr(this));
    bc_arr_ = bc_->ConstructNDArray(&self);
    bn::ndarray & bc_arr = *static_cast<bn::ndarray *>(&bc_arr_);
    bn::dtype bc_dtype = bc_arr.get_dtype();
    // We need to initialize the new bin content values, if the data type
    // is object.
    if(! bn::dtype::equivalent(bc_dtype, bn::dtype::get_builtin<bp::object>()))
        return;

    int const nd = bc_arr.get_nd();
    std::vector<intptr_t> const shape = bc_arr.get_shape_vector();

    intptr_t axis_idx_range_min = 0;
    intptr_t axis_idx_range_max = 0;
    if(n_extra_bins < 0)
    {
        axis_idx_range_min = 0;
        axis_idx_range_max = -n_extra_bins - 1;
    }
    else // n_extra_bins > 0
    {
        axis_idx_range_min = old_shape[axis];
        axis_idx_range_max = shape[axis] - 1;
    }

    bn::flat_iterator<bp::object> bc_iter(bc_arr);
    if(nd == 1)
    {
        // We can just use the flat iterator directly.
        intptr_t axis_idx=axis_idx_range_min;
        bc_iter.advance(axis_idx);
        for(; axis_idx <= axis_idx_range_max; ++axis_idx)
        {
            uintptr_t * obj_ptr_ptr = bc_iter.get_object_ptr_ptr();
            bp::object obj = bc_one_ - bc_one_;
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }
    }
    else
    {
        // Create a strides vector for the iteration.
        std::vector<intptr_t> T(nd);
        T[nd-1] = 1;
        for(intptr_t i=nd-2; i>=0; --i)
        {
            T[i] = shape[i+1]*T[i+1];
        }

        // Calculate the number of iterations (without the iteration axis).
        std::cout << "shape = ";
        intptr_t n_iters = 1;
        for(size_t i=0; i<nd; ++i)
        {
            std::cout << shape[i] << ",";
            if(i != axis) {
                n_iters *= shape[i];
            }
        }
        std::cout << std::endl;
        std::cout << "n_iters = " << n_iters << std::endl<<std::flush;

        intptr_t const last_axis = (nd - 1 == axis ? nd - 2 : nd - 1);
        std::cout << "last_axis = "<< last_axis << std::endl<<std::flush;
        for(intptr_t axis_idx=axis_idx_range_min; axis_idx <= axis_idx_range_max; ++axis_idx)
        {
            std::vector<intptr_t> indices(nd, 0);
            indices[axis] = axis_idx;
            // The iteration follows a matrix. The index pointer p indicates
            // index that needs to be incremented.
            // We need to start from the innermost dimension, unless it is the
            // iteration axis.
            intptr_t p = last_axis;
            for(intptr_t i=0; i<n_iters; ++i)
            {
                intptr_t iteridx = 0;
                for(intptr_t j=nd-1; j>=0; --j)
                {
                    iteridx += indices[j]*T[j];
                }
                std::cout << "iteridx = " << iteridx << std::endl<<std::flush;
                bc_iter.jump_to_iter_index(iteridx);

                uintptr_t * obj_ptr_ptr = bc_iter.get_object_ptr_ptr();
                bp::object obj = bc_one_ - bc_one_;
                *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));

                // Move the index pointer to the next outer-axis if the index
                // of the current axis has reached its maximum. Then increase
                // the index and reset all indices to the right of this
                // increased index to zero. After this operation, the index
                // pointer points to the inner-most axis (excluding the
                // iteration axis).
                while(indices[p] == shape[p]-1)
                {
                    --p;
                    if(p == axis) --p;
                }
                indices[p]++;
                while(p < last_axis)
                {
                    ++p;
                    if(p == axis) ++p;
                    indices[p] = 0;
                }
            }
        }
    }
}

void
ndhist::
handle_struct_array(bp::object const & arr_obj)
{
    bn::dtype dt = bn::dtype::new_builtin<void>();

    std::cout << "dt.itemsize = " << dt.get_itemsize() << std::endl;
    std::cout << "dt.char = " << dt.get_char() << std::endl;
    std::cout << "dt.num = " << dt.get_type_num() << std::endl;
    dt.add_field("x", bn::dtype::get_builtin<double>());
    std::cout << "dt.has_fields() = " << dt.has_fields() << std::endl;
    bp::tuple field_names = dt.get_field_names();
    size_t n = bp::len(field_names);
    for(size_t i=0; i<n; ++i)
    {
        bp::object field_name = field_names[i];
        bp::str field_name_str(field_name);
        std::string name = bp::extract<std::string>(field_name);
        std::cout << "field ["<<i<<"] = " << name << std::endl;
    }
    std::cout << "dt.itemsize = " << dt.get_itemsize() << std::endl;

    /*
    bn::ndarray arr = bn::from_object(arr_obj, 0, 1, bn::ndarray::ALIGNED);
    bn::dtype arr_dt = arr.get_dtype();
    std::cout << "arr.nd = " << arr.get_nd() << std::endl;
    std::cout << "arr_dt.is_flexible() = " << arr_dt.is_flexible() << std::endl;
    std::cout << "arr_dt.has_fields() = " << arr_dt.has_fields() << std::endl;
    std::cout << "arr.get_strides = [";
    std::vector<intptr_t> arr_strides = arr.get_strides_vector();
    for(size_t j=0; j<arr_strides.size(); ++j)
    {
        std::cout << arr_strides[j] << ",";
    }
    std::cout << "]"<< std::endl;

    bp::list field_names = arr_dt.get_field_names();
    size_t n = bp::len(field_names);
    for(size_t i=0; i<n; ++i)
    {
        bp::object field_name = field_names[i];
        bp::str field_name_str(field_name);
        std::string name = bp::extract<std::string>(field_name);
        std::cout << "field ["<<i<<"] = " << name << std::endl;
        std::cout << "field byte offset = " << arr_dt.get_field_byte_offset(field_name_str) << std::endl;
        bn::dtype field_dt = arr_dt.get_field_dtype(field_name_str);
        std::cout << "field_dt.is_flexible() = " << field_dt.is_flexible() << std::endl;
        std::cout << "field_dt.has_fields() = " << field_dt.has_fields() << std::endl;
        std::cout << "field_dt.is_array() = " << field_dt.is_array() << std::endl;
        if(field_dt.is_array())
        {
            bn::dtype field_dt_subdtype(field_dt.get_subdtype());
            std::vector<intptr_t> field_dt_shape = field_dt.get_shape_vector();
            std::cout << "field_dt_subdtype.is_flexible() = " << field_dt_subdtype.is_flexible() << std::endl;
            std::cout << "field_dt_subdtype.has_fields() = " << field_dt_subdtype.has_fields() << std::endl;
            std::cout << "field_dt_subdtype.is_array() = " << field_dt_subdtype.is_array() << std::endl;
            std::cout << "field_dt_subdtype shape = [";
            for(size_t j=0; j<field_dt_shape.size(); ++j)
            {
                std::cout << field_dt_shape[j] << ",";
            }
            std::cout << "]"<< std::endl;
        }
        else
        {
            if(! field_dt.has_fields() && name == "x")
            {
                // The field is a bare data type. So we could iterate over the
                // elements.
                // First we get a ndarray slice for the field and then we
                // iterate over this slice using flat_iterator.
                bn::ndarray field_arr = arr[field_name_str];

                std::vector<intptr_t> strides = field_arr.get_strides_vector();
                std::cout << "x_field_arr.get_strides = [";
                for(size_t j=0; j<strides.size(); ++j)
                {
                    std::cout << strides[j] << ",";
                }
                std::cout << "]"<< std::endl;


                bn::flat_iterator<float> field_arr_iter(field_arr);
                std::cout << "x_field_arr = [";
                for(;field_arr_iter != field_arr_iter.end; ++field_arr_iter)
                {
                    std::cout << *field_arr_iter << ",";
                }
                std::cout << "]"<< std::endl;
            }
            if(! field_dt.has_fields() && name == "y")
            {
                // The field is a bare data type. So we could iterate over the
                // elements.
                // First we get a ndarray slice for the field and then we
                // iterate over this slice using flat_iterator.
                bn::ndarray field_arr = arr[field_name_str];

                std::vector<intptr_t> strides = field_arr.get_strides_vector();
                std::cout << "y_field_arr.get_strides = [";
                for(size_t j=0; j<strides.size(); ++j)
                {
                    std::cout << strides[j] << ",";
                }
                std::cout << "]"<< std::endl;
                std::cout << "sizeof(float) = " << sizeof(float) << std::endl;
                std::cout << "y_field_arr.get_dtype.get_itemsize() = " << field_arr.get_dtype().get_itemsize() << std::endl;
                bn::flat_iterator<double> field_arr_iter(field_arr);
                std::cout << "y_field_arr = [";
                for(;field_arr_iter != field_arr_iter.end; ++field_arr_iter)
                {
                    std::cout << *field_arr_iter << ",";
                }
                std::cout << "]"<< std::endl;
            }
        }
    }
    */
}

}//namespace ndhist
