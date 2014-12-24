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
#if !BOOST_PP_IS_ITERATING

#ifndef NDHIST_NDHIST_HPP_INCLUDED
#define NDHIST_NDHIST_HPP_INCLUDED 1


#include <iostream>
#include <stdint.h>

#include <cstring>
#include <vector>

#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/indexed_iterator.hpp>

#include <ndhist/error.hpp>
#include <ndhist/detail/axis.hpp>
#include <ndhist/detail/limits.hpp>
#include <ndhist/detail/ndarray_storage.hpp>
#include <ndhist/detail/oor_fill_record_stack.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

class ndhist
{
  public:

    /** Constructor for creating a generic shaped histogram with equal or
     *  non-equal sized bins. The shape of the histogram is determined
     *  automatically from the length of the edges (minus 1) arrays.
     *
     *  The axes tuple specifies the different dimensions of the histogram.
     *  - Each tuple entry can either be a single ndarray specifying the bin
     *    edges, or a tuple of the form
     *
     *    (edges_ndarry[, axis_name[, autoscale_front_capacity, autoscale_back_capacity]]).
     *
     *    * The edges_ndarry must be a one-dimensional ndarray objects with N+1
     *      elements in ascending order, beeing the bin edges, where N is the number
     *      of bins for this dimension. The axis_name, if given, must be a str
     *      object specifying the name of the axis. The default name of an axis is
     *      "a{I}", where {I} is the index of the axis starting at zero.
     *    * The extra front and back capacities specify the number of extra
     *      bins in the front and at the back of the axis, respectively.
     *      The default is zero. If one of both values is set to a positive
     *      non-zero value and the bins have equal widths, the the number of
     *      bins of the axis will automatically be scaled, when values are
     *      filled outside the current axis range.
     *      Whenever bins need to be added to an axis, the memory of the bin
     *      content array needs to get reallocated. By having some
     *      extra capacity, the number of reallocations can be reduced.
     *      Thus, the performance does not deterioate as havely as without it.
     *  - The different dimensions can have different edge types, e.g. integer
     *    or float, or any other Python type, i.e. objects.
     *
     *  The dt dtype object defines the data type for the bin contents. For a
     *  histogram this is usually an integer or float type.
     *
     *  In case the bin contents are generic Python objects, the bc_class
     *  argument defines this Python object class and is used to initialize the
     *  bin content array with zeros.
     */
    ndhist(
        bp::tuple const & axes
      , bn::dtype const & dt
      , bp::object const & bc_class = bp::object()
    );

    virtual ~ndhist() {}

    /**
     * @brief Returns the maximal dimensionality of the histogram object, which
     *        is still supported for filling with a tuple of arrays as ndvalue
     *        function argument. Otherwise a structured array needs to be used
     *        as ndvalue argument.
     */
    intptr_t get_max_tuple_fill_nd() const
    {
        return NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND;
    }

    /**
     * @brief Constructs the bin content ndarray for releasing it to Python.
     *        The lifetime of this new object and this ndhist object will be
     *        managed through the BoostNumpy ndarray_accessor_return() policy.
     */
    bn::ndarray
    py_get_bin_content_ndarray();

    /**
     * @brief Returns the ndarray holding the bin edges of the given axis.
     *        Note, that this is always a copy, since the edges are supposed
     *        to be readonly, because a re-edging of an already filled histogram
     *        does not make sense.
     */
    bn::ndarray
    get_edges_ndarray(intptr_t axis=0) const;

    /** Fills a given n-dimension value into the histogram's bin content array.
     *  On the Python side, the *ndvalue* is a numpy object array that might
     *  hold values of different types. The order of these types must match the
     *  types of the bin edges vector.
     */
    void
    fill(bp::object const & ndvalue_obj, bp::object weight_obj);

    //void handle_struct_array(bp::object const & arr_obj);

    inline
    std::vector< boost::shared_ptr<detail::Axis> > &
    get_axes()
    {
        return axes_;
    }

    inline
    detail::ndarray_storage &
    get_bc_storage()
    {
        return *bc_;
    }

    inline
    bn::ndarray &
    GetBCArray()
    {
        return *static_cast<bn::ndarray*>(&bc_arr_);
    }

    inline
    bn::ndarray const &
    GetBCArray() const
    {
        return *static_cast<bn::ndarray const *>(&bc_arr_);
    }

    inline
    int
    get_nd() const
    {
        return GetBCArray().get_nd();
    }

    inline
    bn::dtype
    get_ndvalues_dtype() const
    {
        return ndvalues_dt_;
    }

    inline
    bp::object
    get_one() const
    {
        return bc_one_;
    }

    template <typename BCValueType>
    detail::OORFillRecordStack<BCValueType> &
    get_oor_fill_record_stack()
    {
        return *static_cast< detail::OORFillRecordStack<BCValueType>* >(oor_fill_record_stack_.get());
    }

    template <typename BCValueType>
    bn::indexed_iterator<BCValueType> &
    get_oor_arr_iter(uintptr_t oor_arr_idx)
    {
        return *static_cast<bn::indexed_iterator<BCValueType> *>(oor_arr_iter_vec_[oor_arr_idx].get());
    }

    void
    extend_axes(
        std::vector<intptr_t> const & f_n_extra_bins_vec
      , std::vector<intptr_t> const & b_n_extra_bins_vec
    );

    void
    extend_bin_content_array(
        std::vector<intptr_t> const & f_n_extra_bins_vec
      , std::vector<intptr_t> const & b_n_extra_bins_vec
    );

    template <typename BCValueType>
    void
    extend_oor_arrays(
        std::vector<intptr_t> const & f_n_extra_bins_vec
      , std::vector<intptr_t> const & b_n_extra_bins_vec
    );

    void
    initialize_extended_bin_content_axis(
        intptr_t axis
      , intptr_t f_n_extra_bins
      , intptr_t b_n_extra_bins
    );

  private:
    ndhist()
      : ndvalues_dt_(bn::dtype::new_builtin<void>())
    {};

    void create_oor_arrays(uintptr_t nd, bn::dtype const & bc_dt, bp::object const & bc_class);

    /** The dtype object describing the ndvalues structure. It describes a
     *  structured ndarray with field names, one for each axis of the histogram.
     */
    bn::dtype ndvalues_dt_;

    /** The list of pointers to the Axis object for each dimension.
     */
    std::vector< boost::shared_ptr<detail::Axis> > axes_;

    std::vector<intptr_t> axes_extension_max_fcap_vec_;
    std::vector<intptr_t> axes_extension_max_bcap_vec_;

    /** The bin contents.
     */
    boost::shared_ptr<detail::ndarray_storage> bc_;
    bp::object bc_arr_;

    /** The vector holding n-dimensional ndarrays holding the out-of-range (oor)
     *  bins. We separete the oor bins for different amounts of oor axes for a
     *  fill value, because in case an axis is extended, we don't have to move
     *  oor bin values and can just extend the appropriate arrays.
     *  The index of this vectors determines through a bit mask what axes of a
     *  fill value is out-of-range. A bit set to 0 means the axis is oor and 1
     *  otherwise. The shapes of these n-dim. arrays depend on the number of
     *  oor axes. So for instance for n=3, if all axes are oor, ie. index 0, the
     *  shape is (2,2,2) an represents the corners of the cube. It's 2 because
     *  there are always one underflow and one overflow bin per axis.
     *  Continuing with n=3 and two axes are oor, the shape of the n-dim. array
     *  is (S, 2, 2) where S is the length of the axis that was not oor.
     *  Finally, if two axes are not oor, the shape is (S1, S2, 2) where S1 and
     *  S2 are the lengths of the not-oor axes. Note, that the order of the
     *  dimenions of the n-dim. oor arrays is (by definition) ordered ascending
     *  w.r.t. the dimensions of the histogram, so i.e. x,y if the z-axis is oor
     *  and x,z if the y axis is oor.
     *  In total, there are (2^n - 1) oor arrays in this vector.
     *  The bits of the bit mask are (by definition) ordered from right to left,
     *  i.e. zyx.
     */
    std::vector< boost::shared_ptr<detail::ndarray_storage> > oor_arr_vec_;
    std::vector< boost::shared_ptr<bn::detail::iter_iterator_type> > oor_arr_iter_vec_;

    /** The Python object scalar representation of 1 which will be used for
     *  filling when no weights are specified.
     */
    bp::object bc_one_;

    boost::shared_ptr<detail::OORFillRecordStackBase> oor_fill_record_stack_;

    boost::function<void (ndhist &, bp::object const &, bp::object const &)> fill_fct_;
};

template <typename BCValueType>
void
ndhist::
extend_oor_arrays(
    std::vector<intptr_t> const & f_n_extra_bins_vec
  , std::vector<intptr_t> const & b_n_extra_bins_vec
)
{
    bp::object self(bp::ptr(this));

    uintptr_t const nd = f_n_extra_bins_vec.size();
    uintptr_t min_axis = nd - 1;
    for(uintptr_t i=0; i<nd; ++i)
    {
        if(f_n_extra_bins_vec[i] > 0 || b_n_extra_bins_vec[i] > 0)
        {
            min_axis = std::min(min_axis, i);
        }
    }

    // Calculate the smallest idx including the smallest to-get-extended axis.
    uintptr_t oor_idx = (1 << min_axis);

    // Loop over the oor arrays.
    std::vector<intptr_t> arr_f_n_extra_bins_vec(nd);
    std::vector<intptr_t> arr_b_n_extra_bins_vec(nd);
    std::vector<intptr_t> arr_max_fcap_vec(nd);
    std::vector<intptr_t> arr_max_bcap_vec(nd);
    uintptr_t const n_oor_arr = oor_arr_vec_.size();
    for(; oor_idx < n_oor_arr; ++oor_idx)
    {
        std::bitset<NDHIST_LIMIT_MAX_ND> idx_bset(oor_idx);

        memset(&arr_f_n_extra_bins_vec.front(), 0, nd*sizeof(intptr_t));
        memset(&arr_b_n_extra_bins_vec.front(), 0, nd*sizeof(intptr_t));
        memset(&arr_max_fcap_vec.front(), 0, nd*sizeof(intptr_t));
        memset(&arr_max_bcap_vec.front(), 0, nd*sizeof(intptr_t));
        bool any_extra_bins = false;
        uintptr_t p=0;
        for(uintptr_t i=0; i<nd; ++i)
        {
            if(idx_bset.test(i))
            {
                any_extra_bins |= (f_n_extra_bins_vec[i] > 0 || b_n_extra_bins_vec[i] > 0);
                arr_f_n_extra_bins_vec[p] = f_n_extra_bins_vec[i];
                arr_b_n_extra_bins_vec[p] = b_n_extra_bins_vec[i];
                arr_max_fcap_vec[p] = axes_[i]->extension_max_fcap_;
                arr_max_bcap_vec[p] = axes_[i]->extension_max_bcap_;
                ++p;
            }
        }
        if(! any_extra_bins)
        {
            continue;
        }

        oor_arr_vec_[oor_idx]->extend_axes(
            arr_f_n_extra_bins_vec
          , arr_b_n_extra_bins_vec
          , arr_max_fcap_vec
          , arr_max_bcap_vec
          , &self
        );

        // Replace the indexed iterator for this extended oor array.
        bn::ndarray oor_arr = oor_arr_vec_[oor_idx]->ConstructNDArray(&self);
        oor_arr_iter_vec_[oor_idx] = boost::shared_ptr< bn::indexed_iterator<BCValueType> >(new bn::indexed_iterator<BCValueType>(oor_arr));

        //FIXME
        // If the bin content dtype is object we need to initialize the new
        // elements.




    }
}


}// namespace ndhist

#endif // !NDHIST_NDHIST_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define ND BOOST_PP_ITERATION()

template <>
struct nd_traits<ND>
{
    template<typename BCValueType>
    struct fill_traits
    {
        static
        void
        fill(ndhist & self, bp::object const & ndvalues_obj, bp::object const & weight_obj)
        {
            std::cout << "nd_traits<"<< BOOST_PP_STRINGIZE(ND) <<">::fill_traits<BCValueType>::fill" << std::endl;
            if(! PyTuple_Check(ndvalues_obj.ptr()))
            {
                // The input ndvalues object is not a tuple, so we assume it's a
                // structured array, which will be handled by the
                // generic_nd_traits.
                generic_nd_traits::fill_traits<BCValueType>::fill(self, ndvalues_obj, weight_obj);
                return;
            }

            bp::tuple ndvalues_tuple(ndvalues_obj);

            if(bp::len(ndvalues_tuple) != ND)
            {
                std::stringstream ss;
                ss << "The number of elements (" << bp::len(ndvalues_tuple)
                    << ") in the ndvalues tuple must match "
                    << "the dimensionality (" << BOOST_PP_STRINGIZE(ND)
                    << ") of the histogram.";
                throw ValueError(ss.str());
            }

            // Extract the ndarrays from the tuple for the different axes.
            #define NDHIST_IN_NDARRAY(z, n, data) \
                bn::ndarray BOOST_PP_CAT(ndvalue_arr,n) = bn::from_object(ndvalues_tuple[n], self.get_axes()[n]->get_dtype(), 0, 0, bn::ndarray::ALIGNED);
            BOOST_PP_REPEAT(ND, NDHIST_IN_NDARRAY, ~)
            #undef NDHIST_IN_NDARRAY
            bn::ndarray weight_arr = bn::from_object(weight_obj, bn::dtype::get_builtin<BCValueType>(), 0, 0, bn::ndarray::ALIGNED);

            typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
                    ndvalue_core_shape_t;
            typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
                    weight_core_shape_t;
            typedef bn::dstream::array_definition<ndvalue_core_shape_t, void>
                    ndvalue_arr_def;
            typedef bn::dstream::array_definition<weight_core_shape_t, BCValueType>
                    weight_arr_def;
            #define NDHIST_DEF(z, n, data) BOOST_PP_COMMA_IF(n) data
            typedef bn::dstream::detail::loop_service_arity<ND+1>::loop_service<BOOST_PP_REPEAT(ND, NDHIST_DEF, ndvalue_arr_def) , weight_arr_def>
                    loop_service_t;
            #undef NDHIST_DEF
            #define NDHIST_IN_ARR_SERVICE(z, n, data) \
                bn::dstream::detail::input_array_service<ndvalue_arr_def> BOOST_PP_CAT(ndvalue_arr_service,n)(BOOST_PP_CAT(ndvalue_arr,n));
            BOOST_PP_REPEAT(ND, NDHIST_IN_ARR_SERVICE, ~)
            #undef NDHIST_IN_ARR_SERVICE
            bn::dstream::detail::input_array_service<weight_arr_def> weight_arr_service(weight_arr);
            #define NDHIST_DEF(z, n, data) \
                BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(ndvalue_arr_service,n)
            loop_service_t loop_service(BOOST_PP_REPEAT(ND, NDHIST_DEF, ~), weight_arr_service);
            #undef NDHIST_DEF
            #define NDHIST_DEF(z, n, data) \
                bn::detail::iter_operand BOOST_PP_CAT(ndvalue_arr_iter_op,n)( BOOST_PP_CAT(ndvalue_arr_service,n).get_arr(), bn::detail::iter_operand::flags::READONLY::value, BOOST_PP_CAT(ndvalue_arr_service,n).get_arr_bcr_data() );
            BOOST_PP_REPEAT(ND, NDHIST_DEF, ~)
            #undef NDHIST_DEF
            bn::detail::iter_operand weight_arr_iter_op( weight_arr_service.get_arr(), bn::detail::iter_operand::flags::READONLY::value, weight_arr_service.get_arr_bcr_data() );

            bn::detail::iter_flags_t iter_flags =
                bn::detail::iter::flags::REFS_OK::value // This is needed for the
                                                        // weight, which can be bp::object.
              | bn::detail::iter::flags::EXTERNAL_LOOP::value;
            bn::order_t order = bn::KEEPORDER;
            bn::casting_t casting = bn::NO_CASTING;
            intptr_t buffersize = 0;

            #define NDHIST_DEF(z, n, data) \
                , BOOST_PP_CAT(ndvalue_arr_iter_op,n)
            bn::detail::iter iter(
                  iter_flags
                , order
                , casting
                , loop_service.get_loop_nd()
                , loop_service.get_loop_shape_data()
                , buffersize
                BOOST_PP_REPEAT(ND, NDHIST_DEF, ~)
                , weight_arr_iter_op
            );
            #undef NDHIST_DEF
            iter.init_full_iteration();

            // Create an indexed iterator for the bin content array.
            bn::ndarray bc_arr = self.GetBCArray();
            bn::indexed_iterator<BCValueType> bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);

            // Get a handle to the OOR fill record stack.
            OORFillRecordStack<BCValueType> & oorfrstack = self.get_oor_fill_record_stack<BCValueType>();

            // Do the iteration.
            typedef typename bc_value_traits<BCValueType>::ref_type
                    bc_ref_type;
            std::vector<intptr_t> indices(ND, 0);
            std::vector<intptr_t> relative_indices(ND, 0);
            std::vector<intptr_t> f_n_extra_bins_vec(ND, 0);
            std::vector<intptr_t> b_n_extra_bins_vec(ND, 0);
            std::vector<intptr_t> & bc_fcap = self.get_bc_storage().get_front_capacity_vector();
            std::vector<intptr_t> & bc_bcap = self.get_bc_storage().get_back_capacity_vector();
            bool is_oor;
            uintptr_t oor_arr_idx;
            std::vector<intptr_t> oor_arr_noor_indices(ND);
            std::vector<intptr_t> oor_arr_noor_relavive_indices(ND);
            std::vector<intptr_t> oor_arr_noor_axes_indices(ND);
            std::vector<intptr_t> oor_arr_oor_indices(ND);
            std::vector<intptr_t> oor_arr_oor_relavive_indices(ND);
            std::vector<intptr_t> oor_arr_oor_axes_indices(ND);
            uintptr_t oor_arr_noor_size = 0;
            uintptr_t oor_arr_oor_size = 0;
            bool extend_axes;
            bool reallocation_upon_extension = false;

            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Get the weight scalar from the iterator.
                    bc_ref_type weight = bc_value_traits<BCValueType>::get_value_from_iter(iter, ND);

                    // Fill the scalar into the bin content array.
                    // Get the coordinate of the current ndvalue.
                    extend_axes = false;
                    is_oor = false;
                    oor_arr_idx = 0;
                    oor_arr_noor_size = 0;
                    oor_arr_oor_size = 0;
                    for(size_t i=0; i<ND; ++i)
                    {
                        std::cout << "tuple fill: Get bin idx of axis " << i << " of " << ND << std::endl;
                        boost::shared_ptr<detail::Axis> & axis = self.get_axes()[i];
                        char * ndvalue_ptr = iter.get_data(i);
                        axis::out_of_range_t oor;
                        intptr_t bin_idx = axis->get_bin_index_fct(axis->data_, ndvalue_ptr, &oor);
                        if(oor == axis::OOR_NONE)
                        {

                            std::cout << "normal fill i=" << i << "indices.size()="<<indices.size()
                                      << "relative_indices.size() "<< relative_indices.size()<<std::endl;
                            indices[i] = bin_idx;
                            relative_indices[i] = bin_idx;

                            // Mark this axis as available and save the bin
                            // index in the noor indices vector.
                            oor_arr_idx |= (1<<i);
                            oor_arr_noor_indices[oor_arr_noor_size] = bin_idx;
                            oor_arr_noor_relavive_indices[oor_arr_noor_size] = bin_idx;
                            oor_arr_noor_axes_indices[oor_arr_noor_size] = i;
                            ++oor_arr_noor_size;
                        }
                        else
                        {
                            if(axis->is_extendable())
                            {
                                // Mark this axis as available.
                                oor_arr_idx |= (1<<i);
                                std::cout << "axis is extentable" << std::endl;
                                intptr_t const n_extra_bins = axis->request_extension_fct(axis->data_, ndvalue_ptr, oor);
                                if(oor == axis::OOR_UNDERFLOW)
                                {
                                    indices[i] = 0;
                                    relative_indices[i] = n_extra_bins;

                                    oor_arr_noor_indices[oor_arr_noor_size] = 0;
                                    oor_arr_noor_relavive_indices[oor_arr_noor_size] = n_extra_bins;
                                    oor_arr_noor_axes_indices[oor_arr_noor_size] = i;
                                    ++oor_arr_noor_size;

                                    f_n_extra_bins_vec[i] = std::max(-n_extra_bins, f_n_extra_bins_vec[i]);
                                    reallocation_upon_extension |= (f_n_extra_bins_vec[i] > bc_fcap[i]);
                                }
                                else // oor == axis::OOR_OVERFLOW
                                {
                                    intptr_t const index = axis->get_n_bins_fct(axis->data_) + n_extra_bins - 1;

                                    indices[i] = index;
                                    relative_indices[i] = index;

                                    oor_arr_noor_indices[oor_arr_noor_size] = index;
                                    oor_arr_noor_relavive_indices[oor_arr_noor_size] = index;
                                    oor_arr_noor_axes_indices[oor_arr_noor_size] = i;
                                    ++oor_arr_noor_size;

                                    b_n_extra_bins_vec[i] = std::max(n_extra_bins, b_n_extra_bins_vec[i]);
                                    reallocation_upon_extension |= (b_n_extra_bins_vec[i] > bc_bcap[i]);
                                }

                                extend_axes = true;
                            }
                            else
                            {
                                std::cout << "axis has NO autoscale" << std::endl;
                                // The current value is out-of-range on the
                                // current the axis, which is not extendable.
                                // So mark the axis as oor for this value.
                                is_oor = true;

                                if(oor == axis::OOR_UNDERFLOW)
                                {
                                    oor_arr_oor_indices[oor_arr_oor_size] = 0;
                                    oor_arr_oor_relavive_indices[oor_arr_oor_size] = 0;
                                    oor_arr_oor_axes_indices[oor_arr_oor_size] = i;
                                    ++oor_arr_oor_size;
                                }
                                else // oor == axis::OOR_OVERFLOW
                                {
                                    oor_arr_oor_indices[oor_arr_oor_size] = 1;
                                    oor_arr_oor_relavive_indices[oor_arr_oor_size] = 1;
                                    oor_arr_oor_axes_indices[oor_arr_oor_size] = i;
                                    ++oor_arr_oor_size;
                                }
                            }
                        }
                    }

                    // If the value is out-of-range for any axis and that axis
                    // is extenable we want to cache the value in order to
                    // accumulate a stack of out-of-range values before
                    // extending the axes and the bin content array, what is
                    // very expensive esp. in case of a high dimensional
                    // histogram.
                    if(extend_axes)
                    {
                        std::cout << "extend_axes is true, size="<< oorfrstack.get_size() << std::endl<<std::flush;
                        // Check if an actual reallocation is required,
                        // if not don't fill the stack and just do the
                        // axes extension and mark the value as not
                        // out-of-range.
                        if(reallocation_upon_extension)
                        {
                            std::cout << "reallocation required upon extension " << std::endl<<std::flush;
                            // The value is out-of-range for any extandable axis.
                            // Push it into the cache stack. If it returns ``true``
                            // the stack is full and we need to extent the axes and
                            // fill the cached values in.
                            if(oorfrstack.push_back(
                                is_oor
                              , oor_arr_idx
                              , oor_arr_noor_relavive_indices
                              , oor_arr_noor_axes_indices
                              , oor_arr_noor_size
                              , oor_arr_oor_relavive_indices
                              , oor_arr_oor_axes_indices
                              , oor_arr_oor_size
                              , relative_indices
                              , weight))
                            {
                                std::cout << "The stack is full. Flush it." << std::endl<<std::flush;
                                extend_axes_and_flush_oor_fill_record_stack<BCValueType>(self, f_n_extra_bins_vec, b_n_extra_bins_vec, indices, bc_arr, bc_iter, oorfrstack);
                                reallocation_upon_extension = false;
                            }
                        }
                        else
                        {
                            // No reallocation of memory is required for the
                            // extension, so we just extend the axes.
                            std::cout << "no reallocation required upon extension " << std::endl<<std::flush;
                            self.extend_axes(f_n_extra_bins_vec, b_n_extra_bins_vec);
                            self.extend_bin_content_array(f_n_extra_bins_vec, b_n_extra_bins_vec);
                            self.extend_oor_arrays<BCValueType>(f_n_extra_bins_vec, b_n_extra_bins_vec);
                            bc_arr = self.GetBCArray();
                            bc_iter = bn::indexed_iterator<BCValueType>(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);
                            memset(&f_n_extra_bins_vec.front(), 0, ND*sizeof(intptr_t));
                            memset(&b_n_extra_bins_vec.front(), 0, ND*sizeof(intptr_t));
                        }
                    }

                    // Increase the bin content if all bin indices exist.
                    if(!is_oor)
                    {
                        bc_iter.jump_to(indices);
                        bc_ref_type bc_value = *bc_iter;
                        bc_value += weight;
                    }
                    else
                    {
                        std::cout << "is_oor is true, oor_arr_idx ="<< oor_arr_idx << std::endl;
                        // There is at least one axis that is out-of-range.
                        bn::indexed_iterator<BCValueType> & oor_arr_iter = self.get_oor_arr_iter<BCValueType>(oor_arr_idx);

                        memcpy(&indices[0], &oor_arr_noor_indices[0], oor_arr_noor_size*sizeof(intptr_t));
                        memcpy(&indices[oor_arr_noor_size], &oor_arr_oor_indices[0], oor_arr_oor_size*sizeof(intptr_t));

                        std::cout << "jump to indices, ptr" << std::endl;
                        oor_arr_iter.jump_to(indices);
                        bc_ref_type oor_value = *oor_arr_iter;
                        oor_value += weight;
                    }

                    // Jump to the next fill iteration.
                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());

            // Fill the remaining out-of-range values from the stack.
            if(oorfrstack.get_size() > 0)
            {
                extend_axes_and_flush_oor_fill_record_stack<BCValueType>(self, f_n_extra_bins_vec, b_n_extra_bins_vec, indices, bc_arr, bc_iter, oorfrstack);
            }
        }
    };
};

#undef ND
#else
#if BOOST_PP_ITERATION_FLAGS() == 2

#define ND BOOST_PP_ITERATION()

#define NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(BCDTYPE)                          \
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
        oor_fill_record_stack_ = boost::shared_ptr< detail::OORFillRecordStack<BCDTYPE> >(new detail::OORFillRecordStack<BCDTYPE>(nd, oor_stack_size));\
        fill_fct_ = &detail::nd_traits<ND>::fill_traits<BCDTYPE>::fill;        \
        bc_dtype_supported = true;                                             \
    }

#if ND > 1
else
#endif
if(nd == ND)
{
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(bool)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(int16_t)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(uint16_t)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(int32_t)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(uint32_t)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(int64_t)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(uint64_t)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(float)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(double)
    NDHIST_NDTRAITS_BC_DATA_TYPE_SUPPORT(bp::object)
}

#undef NDHIST_BC_DATA_TYPE_SUPPORT

#undef ND

#endif // BOOST_PP_ITERATION_FLAGS() == 2
#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
