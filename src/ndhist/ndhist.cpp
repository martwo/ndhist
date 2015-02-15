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
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <sstream>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/python/list.hpp>
#include <boost/python/refcount.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/iterators/flat_iterator.hpp>
#include <boost/numpy/iterators/indexed_iterator.hpp>
#include <boost/numpy/iterators/multi_flat_iterator.hpp>
#include <boost/numpy/dstream.hpp>
#include <boost/numpy/utilities.hpp>

#include <ndhist/limits.hpp>
#include <ndhist/ndhist.hpp>
#include <ndhist/axis.hpp>
#include <ndhist/type_support.hpp>
#include <ndhist/detail/axis_index_iter.hpp>
#include <ndhist/detail/bin_iter_value_type_traits.hpp>
#include <ndhist/detail/bin_value.hpp>
#include <ndhist/detail/bin_utils.hpp>
#include <ndhist/detail/limits.hpp>
//#include <ndhist/detail/full_multi_axis_index_iter.hpp>
#include <ndhist/detail/utils.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

namespace detail {

template <typename WeightValueType>
static
void
flush_value_cache(
    ndhist                      & self
  , ValueCache<WeightValueType> & value_cache
  , std::vector<intptr_t> const & f_n_extra_bins_vec
  , uintptr_t const               bc_data_offset
)
{
    intptr_t const nd = self.get_nd();

    // Fill in the cached values.
    char * bin_data_addr;
    intptr_t idx = value_cache.get_size();
    while(idx--)
    {
        typename ValueCache<WeightValueType>::cache_entry_type const & entry = value_cache.get_entry(idx);

        std::vector<intptr_t> const & arr_strides = self.bc_->get_data_strides_vector();
        bin_data_addr = self.bc_->get_data() + bc_data_offset;

        // Translate the relative indices into an absolute
        // data address for the extended bin content array.
        for(intptr_t axis=0; axis<nd; ++axis)
        {
            bin_data_addr += (f_n_extra_bins_vec[axis] + entry.relative_indices[axis]) * arr_strides[axis];
        }

        bin_utils<WeightValueType>::increment_bin(bin_data_addr, entry.weight);
    }

    // Finally, clear the stack.
    value_cache.clear();
}

template <typename BCValueType>
struct iadd_fct_traits
{
    static
    void apply(ndhist & self, ndhist const & other)
    {
        if(! self.is_compatible(other))
        {
            std::stringstream ss;
            ss << "The += operator is only defined for two compatible ndhist "
               << "objects!";
            throw AssertionError(ss.str());
        }

        // Add the bin contents of the two ndhist objects.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_iter_value_type_traits<BCValueType>
                  , bin_iter_value_type_traits<BCValueType>
                >
                multi_iter_t;

        bn::ndarray self_bc_arr = self.bc_->construct_ndarray(self.bc_->get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        bn::ndarray other_bc_arr = other.bc_->construct_ndarray(other.bc_->get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        multi_iter_t bc_it(
            self_bc_arr
          , other_bc_arr
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value);
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value  = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 other_bin_value = multi_value.value_1;
            *self_bin_value.noe_  += *other_bin_value.noe_;
            *self_bin_value.sow_  += *other_bin_value.sow_;
            *self_bin_value.sows_ += *other_bin_value.sows_;
            ++bc_it;
        }
    }
};

template <typename BCValueType>
struct idiv_fct_traits
{
    static
    void apply(ndhist & self, bn::ndarray const & value_arr)
    {
        // Divide the bin contents of the ndhist object with the
        // scalar value.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_iter_value_type_traits<BCValueType>
                  , bn::iterators::single_value<BCValueType>
                >
                multi_iter_t;

        bn::ndarray self_bc_arr = self.bc_->construct_ndarray(self.bc_->get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        multi_iter_t bc_it(
            self_bc_arr
          , const_cast<bn::ndarray &>(value_arr)
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value
        );
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 value          = multi_value.value_1;
            *self_bin_value.sow_  /= value;
            *self_bin_value.sows_ /= value * value;
            ++bc_it;
        }
    }
};

template <typename BCValueType>
struct imul_fct_traits
{
    static
    void apply(ndhist & self, bn::ndarray const & value_arr)
    {
        // Multiply the bin contents of the ndhist object with the
        // scalar value.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_iter_value_type_traits<BCValueType>
                  , bn::iterators::single_value<BCValueType>
                >
                multi_iter_t;

        bn::ndarray self_bc_arr = self.bc_->construct_ndarray(self.bc_->get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        multi_iter_t bc_it(
            self_bc_arr
          , const_cast<bn::ndarray &>(value_arr)
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value
        );
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 value          = multi_value.value_1;
            *self_bin_value.sow_  *= value;
            *self_bin_value.sows_ *= value * value;
            ++bc_it;
        }
    }
};


// template <typename BCValueType>
// struct project_fct_traits
// {
//     static
//     ndhist
//     apply(ndhist const & self, std::set<intptr_t> const & axes)
//     {
//         // Create a ndhist with the dimensions specified by axes.
//         uintptr_t const self_nd = self.get_nd();
//         uintptr_t const proj_nd = axes.size();
//
//         bp::list axis_list;
//         std::set<intptr_t>::const_iterator axes_it = axes.begin();
//         std::set<intptr_t>::const_iterator axes_end = axes.end();
//         for(; axes_it != axes_end; ++axes_it)
//         {
//             axis_list.append(self.get_axis_definition(*axes_it));
//         }
//         bp::tuple axes_tuple(axis_list);
//         ndhist proj(axes_tuple, self.bc_weight_dt_, self.bc_class_);
//
//         full_multi_axis_index_iter proj_idx_iter(proj.bc_->get_shape_vector());
//         full_multi_axis_index_iter self_idx_iter(self.bc_->get_shape_vector());
//
//         // Iterate over *all* the bins (including OOR bins) of the projection.
//         std::vector<intptr_t> proj_fixed_axes_indices(proj_nd, axis::FLAGS_FLOATING_INDEX);
//         std::vector<intptr_t> self_fixed_axes_indices(self_nd, axis::FLAGS_FLOATING_INDEX);
//         bin_value<BCValueType> proj_bin;
//         bin_value<BCValueType> self_bin;
//         proj_idx_iter.init_iteration(proj_fixed_axes_indices);
//         while(! proj_idx_iter.is_end())
//         {
//             std::vector<intptr_t> const & proj_indices = proj_idx_iter.get_indices();
//
//             // Get the proj bin.
//             bin_utils<BCValueType>::get_bin_by_indices(proj, proj_bin, proj_indices);
//
//             // Iterate over all the axes of self which are not fixed through the
//             // current projection indices.
//             axes_it = axes.begin();
//             for(uintptr_t i=0; axes_it != axes_end; ++axes_it, ++i)
//             {
//                 self_fixed_axes_indices[*axes_it] = proj_indices[i];
//             }
//
//             self_idx_iter.init_iteration(self_fixed_axes_indices);
//             while(! self_idx_iter.is_end())
//             {
//                 std::vector<intptr_t> const & self_indices = self_idx_iter.get_indices();
//                 // Get the self bin.
//                 if(self_idx_iter.is_oor_bin()) {
//                     get_oor_bin(self, self_bin, self_idx_iter.get_oor_array_index(), self_indices);
//                 }
//                 else {
//                     get_noor_bin(self, self_bin, self_indices);
//                 }
//
//                 // Add the self bin to the proj bin.
//                 *proj_bin.noe_  += *self_bin.noe_;
//                 *proj_bin.sow_  += *self_bin.sow_;
//                 *proj_bin.sows_ += *self_bin.sows_;
//
//                 self_idx_iter.increment();
//             }
//
//             proj_idx_iter.increment();
//         }
//
//         return proj;
//     }
// };

/**
 * @brief Creates a ND-sized vector of ndarray objects which are views into the
 *     complete (i.e. including under- and overflow bins) bin content array.
 *     So, for example, for the first array, the index of the first dimension
 *     is fixed (either 0 for underflow bins or n_bins for overflow bins), and
 *     the indices of the other dimensions of the first array are not fixed. So
 *     the shape of the first array is (1, n_bins_y+2, n_bins_z+2, ...).
 * @note The returned ndarray object don't have the owndata flag set and have
 *     also no base object set. So they should be regarded as internal objects.
 *     In case they are handed out to the user, their base object needs to be
 *     set.
 */
template <typename BCValueType>
static
std::vector<bn::ndarray>
get_field_axes_oor_ndarrays(
    ndhist const & self
  , ::ndhist::axis::out_of_range_t const oortype
  , size_t const field_idx
)
{
    uintptr_t const nd = self.get_nd();
    std::vector<intptr_t> complete_bc_arr_shape = self.bc_->get_shape_vector();
    std::vector<intptr_t> complete_bc_arr_front_capacity = self.bc_->get_front_capacity_vector();
    std::vector<intptr_t> complete_bc_arr_back_capacity = self.bc_->get_back_capacity_vector();
    // Add the under- and overflow bins of the extendable axes to the shape, and
    // remove them from the front- and back capacities, in order to calculate
    // the data offset and strides correctly.
    for(uintptr_t i=0; i<nd; ++i)
    {
        if(self.axes_[i]->is_extendable())
        {
            complete_bc_arr_shape[i] += 2;
            complete_bc_arr_front_capacity[i] -= 1;
            complete_bc_arr_back_capacity[i] -= 1;
        }
    }

    intptr_t const sub_item_byte_offset = (field_idx == 0 ? 0 : self.bc_->get_dtype().get_fields_byte_offsets()[field_idx]);

    // Allocate vectors for the shape, front and back capacities that are used
    // to construct the views of the returned individual ndarrays.
    std::vector<intptr_t> shape(nd);
    std::vector<intptr_t> front_capacity(nd);
    std::vector<intptr_t> back_capacity(nd);

    bn::dtype const dt = (field_idx == 0 ? self.bc_noe_dt_ : self.bc_weight_dt_);

    std::vector<bn::ndarray> array_vec;
    array_vec.reserve(nd);
    for(uintptr_t i=0; i<nd; ++i)
    {
        for(uintptr_t j=0; j<nd; ++j)
        {
            if(j == i)
            {
                // When we get to the i'th ndarray the axis consists of only one
                // index.
                shape[j] = 1;
                if(oortype == ::ndhist::axis::OOR_UNDERFLOW)
                {
                    front_capacity[j] = complete_bc_arr_front_capacity[j];
                    back_capacity[j] = complete_bc_arr_back_capacity[j] + complete_bc_arr_shape[j] - 1;
                }
                else // oortype == ::ndhist::axis::OOR_OVERFLOW
                {
                    front_capacity[j] = complete_bc_arr_front_capacity[j] + complete_bc_arr_shape[j] - 1;
                    back_capacity[j] = complete_bc_arr_back_capacity[j];
                }
            }
            else
            {
                shape[j] = complete_bc_arr_shape[j];
                front_capacity[j] = complete_bc_arr_front_capacity[j];
                back_capacity[j] = complete_bc_arr_back_capacity[j];
            }
        }

        // Construct the ndarray, that is a view into the bin content array.
        bn::ndarray arr = ndarray_storage::construct_ndarray(*self.bc_, dt, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
        array_vec.push_back(arr);
    }

    return array_vec;
}

/*
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

            size_t const nd = self.get_nd();
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
            bp::object self_obj(bp::ptr(&self));
            bn::ndarray bc_arr = self.bc_->ConstructNDArray(self.bc_weight_dt_, 1, &self_obj);
            bn::iterators::indexed_iterator< bn::iterators::single_value<BCValueType> > bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);

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
            typedef typename bin_utils<BCValueType>::ref_type
                    bc_ref_type;
            std::vector<intptr_t> indices(nd);
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Get the weight scalar from the iterator.
                    bc_ref_type weight = bin_utils<BCValueType>::get_weight_type_value_from_iter(iter, 1);

                    // Fill the scalar into the bin content array.
                    // Get the coordinate of the current ndvalue.
                    bool is_overflown = false;
                    for(size_t i=0; i<nd; ++i)
                    {
                        //std::cout << "Get bin idx of axis " << i << " of " << nd << std::endl;
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
                        bc_iter.jump_to(indices);
                        bc_ref_type bc_value = *bc_iter;
                        bc_value += weight;
                    }

                    // Jump to the next fill iteration.
                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    }; // struct fill_traits
}; // struct generic_nd_traits
*/

template <int nd>
struct specific_nd_traits;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND, <ndhist/ndhist.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail

ndhist::
ndhist(
    bp::tuple const & axes
  , bp::object const & dt
  , bp::object const & bc_class
)
  : nd_(bp::len(axes))
  , ndvalues_dt_(bn::dtype::new_builtin<void>())
  , bc_noe_dt_(bn::dtype::get_builtin<uintptr_t>())
  , bc_weight_dt_(bn::dtype(dt))
  , bc_class_(bc_class)
{
    std::vector<intptr_t> shape(nd_);
    axes_extension_max_fcap_vec_.resize(nd_);
    axes_extension_max_bcap_vec_.resize(nd_);

    // Get the axes of the histogram.
    for(size_t i=0; i<nd_; ++i)
    {
        boost::shared_ptr<Axis> axis = bp::extract< boost::shared_ptr<Axis> >(axes[i]);

        // Set an axis name if it is not specified.
        std::string & axis_name = axis->get_name();
        if(axis_name == "")
        {
            std::stringstream ss_axis_name;
            ss_axis_name << "a" << i;
            axis_name = ss_axis_name.str();
        }

        // Add the axis field to the ndvalues dtype object.
        ndvalues_dt_.add_field(axis_name, axis->get_dtype());

        // Add the bin content shape information for this axis. The number of
        // of bins of an axis include possible under- and overflow bins.
        shape[i] = axis->get_n_bins();

        // Set the extra front and back capacity for this axis if the axis is
        // extendable. The +1 is for the under- or overflow bin, which will be
        // always zero but important to have when making a view to the under-
        // and overflow bin arrays.
        if(axis->is_extendable())
        {
            axes_extension_max_fcap_vec_[i] = axis->get_extension_max_fcap() + 1;
            axes_extension_max_bcap_vec_[i] = axis->get_extension_max_bcap() + 1;
        }
        else
        {
            axes_extension_max_fcap_vec_[i] = 0;
            axes_extension_max_bcap_vec_[i] = 0;
        }

        // Add the axis to the axes vector.
        axes_.push_back(axis);
    }

    // TODO: Make this as an option in the constructor.
    intptr_t value_cache_size = 65536;

    // Create a ndarray for the bin content. Each bin content element consists
    // of three sub-elements: n_entries, sum_of_weights, sum_of_weights_squared.
    bn::dtype bc_dt = bn::dtype::new_builtin<void>();
    bc_dt.add_field("n",    bc_noe_dt_);
    bc_dt.add_field("sow",  bc_weight_dt_);
    bc_dt.add_field("sows", bc_weight_dt_);
    bc_ = boost::shared_ptr<detail::ndarray_storage>(new detail::ndarray_storage(shape, axes_extension_max_fcap_vec_, axes_extension_max_bcap_vec_, bc_dt));

    // Set the fill function based on the bin content data type.
    bool bc_dtype_supported = false;
    #define BOOST_PP_ITERATION_PARAMS_1                                        \
        (4, (1, NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND, <ndhist/ndhist.hpp>, 2))
    #include BOOST_PP_ITERATE()
    else
    {
        // nd is greater than NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND.
        #define NDHIST_WEIGHT_VALUE_TYPE_SUPPORT(r, data, BCDTYPE)              \
            if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<BCDTYPE>()))\
            {                                                                   \
                if(bc_dtype_supported)                                          \
                {                                                               \
                    std::stringstream ss;                                       \
                    ss << "The bin content data type is supported by more than "\
                       << "one possible C++ data type! This is an internal "    \
                       << "error!";                                             \
                    throw TypeError(ss.str());                                  \
                }                                                               \
                value_cache_ = boost::shared_ptr< detail::ValueCache<BCDTYPE> >(new detail::ValueCache<BCDTYPE>(nd_, value_cache_size));\
                /*fill_fct_ = &detail::generic_nd_traits::fill_traits<BCDTYPE>::fill;*/\
                bc_dtype_supported = true;                                      \
            }
        BOOST_PP_SEQ_FOR_EACH(NDHIST_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)
        #undef NDHIST_WEIGHT_VALUE_TYPE_SUPPORT
    }
    if(!bc_dtype_supported)
    {
        std::stringstream ss;
        ss << "The data type of the bin content array is not supported.";
        throw TypeError(ss.str());
    }

    // Setup the function pointers, which depend on the weight value type.
    #define NDHIST_WEIGHT_VALUE_TYPE_SUPPORT(r, data, WEIGHT_VALUE_TYPE)        \
        if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<WEIGHT_VALUE_TYPE>()))\
        {                                                                       \
            iadd_fct_ = &detail::iadd_fct_traits<WEIGHT_VALUE_TYPE>::apply;     \
            idiv_fct_ = &detail::idiv_fct_traits<WEIGHT_VALUE_TYPE>::apply;     \
            imul_fct_ = &detail::imul_fct_traits<WEIGHT_VALUE_TYPE>::apply;     \
            get_weight_type_field_axes_oor_ndarrays_fct_ = &detail::get_field_axes_oor_ndarrays<WEIGHT_VALUE_TYPE>;\
            /*project_fct_ = &detail::project_fct_traits<WEIGHT_VALUE_TYPE>::apply;*/\
        }
    BOOST_PP_SEQ_FOR_EACH(NDHIST_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)
    #undef NDHIST_WEIGHT_VALUE_TYPE_SUPPORT

    get_noe_type_field_axes_oor_ndarrays_fct_ = &detail::get_field_axes_oor_ndarrays<uintptr_t>;

    // Initialize the bin content array with objects using their default
    // constructor when the bin content array is an object array.
    if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
    {
        bn::ndarray bc_arr = construct_complete_bin_content_ndarray(bc_->get_dtype());
        bn::iterators::flat_iterator< detail::bin_iter_value_type_traits<bp::object> > bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);
        while(! bc_iter.is_end())
        {
            detail::bin_iter_value_type_traits<bp::object>::value_ref_type bin = *bc_iter;

            bp::object sow_obj  = bc_class_();
            bp::object sows_obj = bc_class_();
            *bin.sow_obj_ptr_  = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sow_obj.ptr()));
            *bin.sows_obj_ptr_ = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sows_obj.ptr()));

            ++bc_iter;
        }
    }
}

ndhist::
~ndhist()
{
    // If the bin content array is an object array, we need to decref the
    // objects we have created (and incref'ed) at construction time.
    if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
    {
        bn::ndarray bc_arr = construct_complete_bin_content_ndarray(bc_->get_dtype());
        bn::iterators::flat_iterator< detail::bin_iter_value_type_traits<bp::object> > bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);
        while(! bc_iter.is_end())
        {
            detail::bin_iter_value_type_traits<bp::object>::value_ref_type bin = *bc_iter;

            bp::decref<PyObject>(reinterpret_cast<PyObject*>(*bin.sow_obj_ptr_));
            bp::decref<PyObject>(reinterpret_cast<PyObject*>(*bin.sows_obj_ptr_));
            ++bc_iter;
        }
    }
}

ndhist &
ndhist::operator+=(ndhist const & rhs)
{
    iadd_fct_(*this, rhs);
    return *this;
}

ndhist
ndhist::
operator+(ndhist const & rhs) const
{
    if(! this->is_compatible(rhs))
    {
        std::stringstream ss;
        ss << "The right-hand-side ndhist object for the + operator must be "
           << "compatible with this ndhist object!";
        throw AssertionError(ss.str());
    }

    ndhist newhist = this->empty_like();
    newhist += *this;
    newhist += rhs;

    return newhist;
}

/*
ndhist
ndhist::
operator[](bp::object dim_slices) const
{
    // TODO: convert single object to list of length 1 first.
    if(! PyTuple_Check(dim_slices.ptr()))
    {
        dim_slices = bp::make_tuple(dim_slices);
    }

    // Determine the bin edges arrays of the new histogram.
    uintptr_t const dim = bp::len(dim_slices);
    if(dim > nd_)
    {
        std::stringstream ss;
        ss << "The number of given slices must not exceed "<< nd_ << "!";
        throw IndexError(ss.str());
    }

    if(dim < nd_)
    {
        // If all specified axes are intergers, the user has selected a certain
        // sub-histogram of this histogram which those indices fixed and full
        // axis range for the remaining axes.

        // But if one of 
    }

    std::cout << "Got "<< dim << " axes indices." << std::endl;
    bp::list newaxes_list;
    for(uintptr_t i=0; i<dim; ++i)
    {
        bp::object dim_slice_obj = dim_slices[i];
        if(bn::is_any_scalar(dim_slice_obj))
        {
            intptr_t const start = bp::extract<intptr_t>(dim_slice_obj);
            std::cout << "Got scalar '"<<start<<"' for axis '"<<i<<"'"<<std::endl;
            dim_slice_obj = bp::slice(start, start+1, 1);
        }

        if(detail::py::are_same_type_objects(dim_slice_obj, bp::slice()))
        {
            // Note: The dim_slice_obj is supposed to include the underflow (0) and
            // overflow bin (nbin) but the edges array does not have an explicit
            // underflow (-inf) and overflow (+inf) edge.
            uintptr_t const axis_size = axes_[i]->get_n_bins_fct(axes_[i]->data_);

            bp::slice axis_slice = bp::extract<bp::slice>(dim_slice_obj);
            detail::axis_index_iter axis_idx_iter(axis_size, detail::axis::UNDERFLOW_INDEX, detail::axis::OVERFLOW_INDEX);
            detail::axis_index_iter axis_idx_iter_end = axis_idx_iter.end();
            bp::slice::range<detail::axis_index_iter> axis_iter_range = axis_slice.get_indices(axis_idx_iter, axis_idx_iter_end);

            std::cout << "axis="<<i<<", sliced indices=";
            while(axis_iter_range.start != axis_iter_range.stop)
            {
                intptr_t idx = *axis_iter_range.start;
                if(idx == detail::axis::UNDERFLOW_INDEX) {
                    std::cout << "U,";
                }
                else if(idx == detail::axis::OVERFLOW_INDEX) {
                    std::cout << "O,";
                }
                else {
                    std::cout << idx << ",";
                }
                std::advance(axis_iter_range.start, axis_iter_range.step);
            }
            intptr_t idx = *axis_iter_range.start;
            if(idx == detail::axis::UNDERFLOW_INDEX) {
                std::cout << "U,";
            }
            else if(idx == detail::axis::OVERFLOW_INDEX) {
                std::cout << "O,";
            }
            else {
                std::cout << idx << ",";
            }

            std::cout << std::endl << std::flush;
        }
        else if(bn::is_ndarray(dim_slice_obj))
        {
            throw TypeError("ndarray indexing is not supported yet.");
        }
        else
        {
            std::stringstream ss;
            ss << "The dimension slices must be given either as integer or "
               << "slice objects!";
            throw IndexError(ss.str());
        }





        // Create the axis definition for the new sub-axis.
//         bn::ndarray edges_arr = axes_[i]->get_edges_ndarray_fct(axes_[i]->data_);
//         bn::ndarray subedges_arr = edges_arr[dim_slice_obj];
//         newaxes_list.append(get_axis_definition(i, subedges_arr));
    }
    bp::tuple newaxes(newaxes_list);
    return empty_like();
}
*/

bool
ndhist::
is_compatible(ndhist const & other) const
{
    if(nd_ != other.nd_) {
        return false;
    }
    for(uintptr_t i=0; i<nd_; ++i)
    {
        bn::ndarray const this_axis_edges_arr = this->axes_[i]->get_edges_ndarray();
        bn::ndarray const other_axis_edges_arr = other.axes_[i]->get_edges_ndarray();

        if(bn::all(bn::equal(this_axis_edges_arr, other_axis_edges_arr), 0) == false)
        {
            return false;
        }
    }

    return true;
}

ndhist
ndhist::
empty_like() const
{
    bp::list axis_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        axis_list.append(axes_[i]);
    }
    bp::tuple axes(axis_list);
    return ndhist(axes, bc_weight_dt_, bc_class_);
}

// ndhist
// ndhist::
// project(bp::object const & dims) const
// {
//     intptr_t const nd = get_nd();
//     bn::ndarray axes_arr = bn::from_object(dims, bn::dtype::get_builtin<intptr_t>(), 0, 1, bn::ndarray::ALIGNED);
//     std::set<intptr_t> axes;
//     bn::iterators::flat_iterator< bn::iterators::single_value<intptr_t> > axes_arr_iter(axes_arr);
//     while(! axes_arr_iter.is_end())
//     {
//         intptr_t axis = *axes_arr_iter;
//         if(axis < 0) {
//             axis += nd;
//         }
//         if(axis < 0)
//         {
//             std::stringstream ss;
//             ss << "The axis value \""<< *axes_arr_iter <<"\" specifies an "
//                << "axis < 0!";
//             throw IndexError(ss.str());
//         }
//         else if(axis >= nd)
//         {
//             std::stringstream ss;
//             ss << "The axis value \""<< axis <<"\" must be smaller than the "
//                << "dimensionality of the histogram, i.e. smaller than "
//                << nd <<"!";
//             throw IndexError(ss.str());
//         }
//         if(! axes.insert(axis).second)
//         {
//             std::stringstream ss;
//             ss << "The axis value \""<< axis <<"\" has been "
//                << "specified at least twice!";
//             throw ValueError(ss.str());
//         }
//         ++axes_arr_iter;
//     }
//     return project_fct_(*this, axes);
// }

bp::tuple
ndhist::
py_get_nbins() const
{
    std::vector<intptr_t> const & shape = bc_->get_shape_vector();
    std::vector<intptr_t> nbins(nd_);
    for(uintptr_t i=0; i<nd_; ++i)
    {
        nbins[i] = (axes_[i]->is_extendable() ? shape[i] : shape[i]-2);
    }
    bp::list shape_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        shape_list.append(nbins[i]);
    }
    bp::tuple shape_tuple(shape_list);
    return shape_tuple;
}

bn::ndarray
ndhist::
py_get_noe_ndarray()
{
    // The core part of the bin content array excludes the under- and
    // overflow bins. So we need to create an appropriate view into the bin
    // content array.
    std::vector<intptr_t> shape;
    std::vector<intptr_t> front_capacity;
    std::vector<intptr_t> back_capacity;
    calc_core_bin_content_ndarray_settings(shape, front_capacity, back_capacity);

    intptr_t const sub_item_byte_offset = 0;

    return detail::ndarray_storage::construct_ndarray(*bc_, bc_noe_dt_, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
}

bn::ndarray
ndhist::
py_get_sow_ndarray()
{
    // The core part of the bin content array excludes the under- and
    // overflow bins. So we need to create an appropriate view into the bin
    // content array.
    std::vector<intptr_t> shape;
    std::vector<intptr_t> front_capacity;
    std::vector<intptr_t> back_capacity;
    calc_core_bin_content_ndarray_settings(shape, front_capacity, back_capacity);

    intptr_t const sub_item_byte_offset = bc_->get_dtype().get_fields_byte_offsets()[1];

    return detail::ndarray_storage::construct_ndarray(*bc_, bc_weight_dt_, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
}

bn::ndarray
ndhist::
py_get_sows_ndarray()
{
    // The core part of the bin content array excludes the under- and
    // overflow bins. So we need to create an appropriate view into the bin
    // content array.
    std::vector<intptr_t> shape;
    std::vector<intptr_t> front_capacity;
    std::vector<intptr_t> back_capacity;
    calc_core_bin_content_ndarray_settings(shape, front_capacity, back_capacity);

    intptr_t const sub_item_byte_offset = bc_->get_dtype().get_fields_byte_offsets()[2];

    return detail::ndarray_storage::construct_ndarray(*bc_, bc_weight_dt_, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
}

bp::tuple
ndhist::
py_get_labels() const
{
    bp::list labels_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        labels_list.append(axes_[i]->get_label());
    }
    bp::tuple labels_tuple(labels_list);
    return labels_tuple;
}

bp::tuple
ndhist::
py_get_underflow_entries() const
{
    std::vector<bn::ndarray> array_vec = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 0);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        // TODO: Right now we use the copy method to make the copy. But we
        //       should use deepcopy (which still needs to be implemented in
        //       BoostNumpy) to also copy the objects in an object array.
        underflow_list.append(array_vec[i].copy());
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_underflow_entries_view() const
{
    std::vector<bn::ndarray> array_vec = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 0);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        underflow_list.append(array_vec[i]);
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_overflow_entries() const
{
    std::vector<bn::ndarray> array_vec = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 0);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        // TODO: Right now we use the copy method to make the copy. But we
        //       should use deepcopy (which still needs to be implemented in
        //       BoostNumpy) to also copy the objects in an object array.
        overflow_list.append(array_vec[i].copy());
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}

bp::tuple
ndhist::
py_get_overflow_entries_view() const
{
    std::vector<bn::ndarray> array_vec = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 0);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        overflow_list.append(array_vec[i]);
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}

bp::tuple
ndhist::
py_get_underflow() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 1);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        // TODO: Right now we use the copy method to make the copy. But we
        //       should use deepcopy (which still needs to be implemented in
        //       BoostNumpy) to also copy the objects in an object array.
        underflow_list.append(array_vec[i].copy());
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_underflow_view() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 1);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        underflow_list.append(array_vec[i]);
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_overflow() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 1);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        // TODO: Right now we use the copy method to make the copy. But we
        //       should use deepcopy (which still needs to be implemented in
        //       BoostNumpy) to also copy the objects in an object array.
        overflow_list.append(array_vec[i].copy());
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}

bp::tuple
ndhist::
py_get_overflow_view() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 1);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        overflow_list.append(array_vec[i]);
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}

bp::tuple
ndhist::
py_get_underflow_squaredweights() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 2);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        // TODO: Right now we use the copy method to make the copy. But we
        //       should use deepcopy (which still needs to be implemented in
        //       BoostNumpy) to also copy the objects in an object array.
        underflow_list.append(array_vec[i].copy());
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_underflow_squaredweights_view() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 2);

    bp::list underflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        underflow_list.append(array_vec[i]);
    }
    bp::tuple underflow(underflow_list);
    return underflow;
}

bp::tuple
ndhist::
py_get_overflow_squaredweights() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 2);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        // TODO: Right now we use the copy method to make the copy. But we
        //       should use deepcopy (which still needs to be implemented in
        //       BoostNumpy) to also copy the objects in an object array.
        overflow_list.append(array_vec[i].copy());
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}

bp::tuple
ndhist::
py_get_overflow_squaredweights_view() const
{
    std::vector<bn::ndarray> array_vec = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 2);

    bp::list overflow_list;
    for(size_t i=0; i<array_vec.size(); ++i)
    {
        overflow_list.append(array_vec[i]);
    }
    bp::tuple overflow(overflow_list);
    return overflow;
}

bn::ndarray
ndhist::
get_edges_ndarray(intptr_t axis) const
{
    // Count axis from the back if axis is negative.
    if(axis < 0) {
        axis += axes_.size();
    }

    if(axis < 0 || axis >= intptr_t(axes_.size()))
    {
        std::stringstream ss;
        ss << "The axis parameter must be in the interval "
           << "[0, " << axes_.size()-1 << "] or "
           << "[-1, -"<< axes_.size() <<"]!";
        throw IndexError(ss.str());
    }

    return axes_[axis]->get_edges_ndarray();
}

bp::object
ndhist::
py_get_binedges() const
{
    // Special case for 1d histograms, where we skip the extra tuple of
    // length 1.
    if(nd_ == 1)
    {
        return get_edges_ndarray(0);
    }

    bp::list edges_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        bn::ndarray edges = get_edges_ndarray(i);
        edges_list.append(edges);
    }
    bp::tuple edges_tuple(edges_list);
    return edges_tuple;
}

void
ndhist::
py_fill(bp::object const & ndvalue_obj, bp::object weight_obj)
{
    // In case None is given as weight, we will use one.
    if(weight_obj == bp::object())
    {
        weight_obj = bp::object(1);
    }
    fill_fct_(*this, ndvalue_obj, weight_obj);
}

// TODO: Use the new multi_axis_iter for this.
static
void
initialize_extended_array_axis_range(
    bn::iterators::flat_iterator< bn::iterators::single_value<bp::object> > & iter
  , intptr_t axis
  , std::vector<intptr_t> const & shape
  , std::vector<intptr_t> const & strides
  , intptr_t n_iters
  , intptr_t axis_idx_range_min
  , intptr_t axis_idx_range_max
  , bp::object const obj_class
)
{
    int const nd = strides.size();

    intptr_t const last_axis = (nd - 1 == axis ? nd - 2 : nd - 1);
    //std::cout << "last_axis = "<< last_axis << std::endl<<std::flush;
    std::vector<intptr_t> indices(nd);
    for(intptr_t axis_idx=axis_idx_range_min; axis_idx < axis_idx_range_max; ++axis_idx)
    {
        //std::cout << "Start new axis idx"<<std::endl<<std::flush;
        memset(&indices.front(), 0, nd*sizeof(intptr_t));
        indices[axis] = axis_idx;
        // The iteration follows a matrix. The index pointer p indicates
        // index that needs to be incremented.
        // We need to start from the innermost dimension, unless it is the
        // iteration axis.
        intptr_t p = last_axis;
        for(intptr_t i=0; i<n_iters; ++i)
        {
            //std::cout << "indices = ";
            //for(intptr_t j=0; j<nd; ++j)
            //{
                //std::cout << indices[j] << ",";
            //}
            //std::cout << std::endl;

            intptr_t iteridx = 0;
            for(intptr_t j=nd-1; j>=0; --j)
            {
                iteridx += indices[j]*strides[j];
            }
            //std::cout << "iteridx = " << iteridx << std::endl<<std::flush;
            iter.jump_to_iter_index(iteridx);
            //std::cout << "jump done" << std::endl<<std::flush;

            *iter;
            uintptr_t * obj_ptr_ptr = iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            //std::cout << "Setting pointer data ..."<<std::flush;
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
            //std::cout << "done."<<std::endl<<std::flush;
            if(i == n_iters-1) break;
            // Move the index pointer to the next outer-axis if the index
            // of the current axis has reached its maximum. Then increase
            // the index and reset all indices to the right of this
            // increased index to zero. After this operation, the index
            // pointer points to the inner-most axis (excluding the
            // iteration axis).
            //std::cout << "p1 = "<<p<<std::endl<<std::flush;
            while(indices[p] == shape[p]-1)
            {
                --p;
                if(p == axis) --p;
            }
            //std::cout << "p2 = "<<p<<std::endl<<std::flush;
            indices[p]++;
            while(p < last_axis)
            {
                ++p;
                if(p == axis) ++p;
                indices[p] = 0;
            }

            //std::cout << "p3 = "<<p<<std::endl;
        }
    }
}

void
ndhist::
initialize_extended_array_axis(
    bp::object & arr_obj
  , bp::object const & obj_class
  , intptr_t axis
  , intptr_t f_n_extra_bins
  , intptr_t b_n_extra_bins
)
{
    if(f_n_extra_bins == 0 && b_n_extra_bins == 0) return;

    bn::ndarray & arr = *static_cast<bn::ndarray *>(&arr_obj);
    std::vector<intptr_t> const shape = arr.get_shape_vector();

    intptr_t f_axis_idx_range_min = 0;
    intptr_t f_axis_idx_range_max = 0;
    if(f_n_extra_bins > 0)
    {
        f_axis_idx_range_min = 0;
        f_axis_idx_range_max = f_n_extra_bins;
    }

    intptr_t b_axis_idx_range_min = 0;
    intptr_t b_axis_idx_range_max = 0;
    if(b_n_extra_bins > 0)
    {
        b_axis_idx_range_min = shape[axis] - b_n_extra_bins;
        b_axis_idx_range_max = shape[axis];
    }

    bn::iterators::flat_iterator< bn::iterators::single_value<bp::object> > bc_iter(arr, bn::detail::iter_operand::flags::WRITEONLY::value);
    intptr_t const nd = arr.get_nd();
    if(nd == 1)
    {
        // We can just use the flat iterator directly.

        // --- for front elements.
        for(intptr_t axis_idx = f_axis_idx_range_min; axis_idx < f_axis_idx_range_max; ++axis_idx)
        {
            bc_iter.jump_to_iter_index(axis_idx);
            *bc_iter;
            uintptr_t * obj_ptr_ptr = bc_iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }

        // --- for back elements.
        for(intptr_t axis_idx = b_axis_idx_range_min; axis_idx < b_axis_idx_range_max; ++axis_idx)
        {
            bc_iter.jump_to_iter_index(axis_idx);
            *bc_iter;
            uintptr_t * obj_ptr_ptr = bc_iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }
    }
    else
    {
        // Create a strides vector for the iteration.
        std::vector<intptr_t> strides(nd);
        strides[nd-1] = 1;
        for(intptr_t i=nd-2; i>=0; --i)
        {
            strides[i] = shape[i+1]*strides[i+1];
        }

        // Calculate the number of iterations (without the iteration axis).
        //std::cout << "shape = ";
        intptr_t n_iters = 1;
        for(intptr_t i=0; i<nd; ++i)
        {
            //std::cout << shape[i] << ",";
            if(i != axis) {
                n_iters *= shape[i];
            }
        }
        //std::cout << std::endl;
        //std::cout << "n_iters = " << n_iters << std::endl<<std::flush;

        // Initialize the front elements.
        initialize_extended_array_axis_range(bc_iter, axis, shape, strides, n_iters, f_axis_idx_range_min, f_axis_idx_range_max, obj_class);
        // Initialize the back elements.
        initialize_extended_array_axis_range(bc_iter, axis, shape, strides, n_iters, b_axis_idx_range_min, b_axis_idx_range_max, obj_class);
    }
}

void
ndhist::
extend_axes(
    std::vector<intptr_t> const & f_n_extra_bins_vec
  , std::vector<intptr_t> const & b_n_extra_bins_vec
)
{
    for(uintptr_t i=0; i<nd_; ++i)
    {
        Axis & axis = *this->axes_[i];
        if(axis.is_extendable())
        {
            axis.extend(f_n_extra_bins_vec[i], b_n_extra_bins_vec[i]);
        }
    }
}

void
ndhist::
extend_bin_content_array(
    std::vector<intptr_t> const & f_n_extra_bins_vec
  , std::vector<intptr_t> const & b_n_extra_bins_vec
)
{
    // Extend the bin content array. This might cause a reallocation of memory.
    bc_->extend_axes(f_n_extra_bins_vec, b_n_extra_bins_vec, axes_extension_max_fcap_vec_, axes_extension_max_bcap_vec_);

    // We need to initialize the new bin content values, if the data type
    // is object.
    if(! bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
        return;

    std::vector<intptr_t> f_n_extra_bins = f_n_extra_bins_vec;
    std::vector<intptr_t> b_n_extra_bins = b_n_extra_bins_vec;

    bn::ndarray bc_sow_arr  = construct_complete_bin_content_ndarray(bc_weight_dt_, 1);
    bn::ndarray bc_sows_arr = construct_complete_bin_content_ndarray(bc_weight_dt_, 2);
    for(uintptr_t axis=0; axis<nd_; ++axis)
    {
        // In case the axis is extendable we also need to initialize the shifted
        // virtual under and overflow bin.
        if(axes_[axis]->is_extendable())
        {
            if(f_n_extra_bins[axis] > 0) { f_n_extra_bins[axis] += 1; }
            if(b_n_extra_bins[axis] > 0) { b_n_extra_bins[axis] += 1; }
        }
        initialize_extended_array_axis(bc_sow_arr,  bc_class_, axis, f_n_extra_bins[axis], b_n_extra_bins[axis]);
        initialize_extended_array_axis(bc_sows_arr, bc_class_, axis, f_n_extra_bins[axis], b_n_extra_bins[axis]);
    }
}

bn::ndarray
ndhist::
construct_complete_bin_content_ndarray(
    bn::dtype const & dt
  , size_t const field_idx
) const
{
    std::vector<intptr_t> shape = bc_->get_shape_vector();
    std::vector<intptr_t> front_capacity = bc_->get_front_capacity_vector();
    std::vector<intptr_t> back_capacity = bc_->get_back_capacity_vector();
    // Add the under- and overflow bins of the extendable axes to the shape, and
    // remove them from the front- and back capacities, in order to calculate
    // the data offset and strides correctly.
    for(uintptr_t i=0; i<nd_; ++i)
    {
        if(axes_[i]->is_extendable())
        {
            shape[i] += 2;
            front_capacity[i] -= 1;
            back_capacity[i] -= 1;
        }
    }

    intptr_t const sub_item_byte_offset = (field_idx == 0 ? 0 : bc_->get_dtype().get_fields_byte_offsets()[field_idx]);

    return detail::ndarray_storage::construct_ndarray(*bc_, dt, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
}

}//namespace ndhist
