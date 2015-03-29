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

#include <stdint.h>

#include <cstring>
#include <iostream>
#include <set>
#include <vector>

#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/python.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>

#include <ndhist/axis.hpp>
#include <ndhist/error.hpp>
#include <ndhist/detail/limits.hpp>
#include <ndhist/detail/ndarray_storage.hpp>
#include <ndhist/detail/value_cache.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

class ndhist
  : public boost::enable_shared_from_this<ndhist>
{
  public:

    /**
     * @brief Constructor for creating a generic shaped histogram with equal or
     *  non-equal sized bins. The shape of the histogram is determined
     *  automatically from the given axis objects.
     *
     *  The axes tuple specifies the different dimensions of the histogram.
     *  Each element of that tuple is supposed to be an object of (a derived)
     *  class of ndhist::Axis.
     *  - Each tuple entry can either be a single ndarray specifying the bin
     *    edges, or a tuple of the form
     *  - The different dimensions can have different edge types, e.g. integer
     *    or float, or any other Python type, i.e. objects.
     *
     *  The dt dtype object defines the data type for a weight value. For a
     *  histogram this is usually an integer or floating point type.
     *
     *  In case the bin contents are generic Python objects, the bc_class
     *  argument defines this Python object class and is used to initialize the
     *  bin content array with zeros.
     */
    ndhist(
        bp::tuple const & axes
      , bp::object const & dt
      , bp::object const & bc_class = bp::object()
    );

    /**
     * @brief Constructs a new ndhist object, that shares the bin content array
     *     with the given owner ndhist object. The given axes define the axes
     *     of the new ndhist object. The data_offset, data_shape, and
     *     data_strides constants define the data view into the bin content
     *     array. The lengths of data_shape and data_strides vectors must match.
     *     They define the dimensionality of the new ndhist histogram.
     */
    ndhist(
        ndhist const & base
      , std::vector< boost::shared_ptr<Axis> > const & axes
      , intptr_t const bytearray_data_offset
      , std::vector<intptr_t> const & data_shape
      , std::vector<intptr_t> const & data_strides
    );

    virtual
    ~ndhist();

    /**
     * @brief Copies this ndhist object. If this ndhist object is a view, the
     *     resulting ndhist object is also a view and the internal bytearray is
     *     not copied.
     */
    ndhist
    copy() const;

    // Operator overloads.
    /**
     * @brief Adds the given right-hand-side histogram to this ndhist object and
     *        returning a reference to this (altered) ndhist object.
     *        The two histograms need to be compatible to each other.
     */
    ndhist & operator+=(ndhist const & rhs);

    /**
     * @brief Scales (multiplies) the sum of weights and the sum of weights
     *        squared of this histogram by the given scalar value.
     */
    template <typename T>
    ndhist & operator*=(T const & rhs);

    /**
     * @brief Scales (divides) the sum of weights and the sum of weights
     *        squared of this histogram by the given scalar value.
     */
    template <typename T>
    ndhist & operator/=(T const & rhs);

    /**
     * @brief Implements the operation ``ndhist = *this + rhs``.
     */
    ndhist operator+(ndhist const & rhs) const;

    /**
     * @brief Implements the operation ``ndhist = *this * rhs``.
     */
    template <typename T>
    ndhist operator*(T const & rhs) const;

    /**
     * @brief Implements the operation ``ndhist = *this / rhs``.
     */
    template <typename T>
    ndhist operator/(T const & rhs) const;

    /**
     * @brief Gets a "sub"-histogram of this histogram specified through bin
     *     indexing of this histogram.
     *     According to the indexing documentation of numpy [1] we support
     *     basic indexing, i.e. the returned (sub) histogram is a view into this
     *     histogram object.
     *
     *     [1] http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
     */
    ndhist operator[](bp::object const & arg) const;

    /**
     * @brief Checks if the given ndhist object is compatible with this ndhist
     *        object. This means, the dimensionality, and the bin edges must
     *        match exactly between the two.
     */
    bool
    is_compatible(ndhist const & other) const;

    /**
     * @brief Creates a new empty ndhist object that has the same binning as
     *        this histogram.
     */
    ndhist
    empty_like() const;

    /**
     * @brief Returns the maximal dimensionality of the histogram object, which
     *        is still supported for filling with a tuple of arrays as ndvalue
     *        function argument. Otherwise a structured ndarray needs to be used
     *        as ndvalue argument.
     */
    intptr_t
    get_max_tuple_fill_nd() const
    {
        return NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND;
    }

    /**
     * @brief Create a boost::python::tuple object holding the shape of the
     *     histogram. The shape numbers are the number of bins for each axis,
     *     where possible under- and overflow bins are included.
     */
    bp::tuple
    py_get_shape() const;

    /**
     * @brief Creates a boost::python::tuple object holding the number of bins
     *        for each axis.
     */
    bp::tuple
    py_get_nbins() const;

    /**
     * @brief Creates a boost::python::tuple object holding the Axis objects
     *     of this histogram.
     */
    bp::tuple
    py_get_axes() const;

    /**
     * @brief Constructs the number of entries ndarray for releasing it to
     *     Python.
     *     The lifetime of this new object and this ndhist object will be
     *     managed through the BoostNumpy ndarray_accessor_return() policy.
     * @note In case this ndhist is a zero-dimensional histogram this method
     *     returns a scalar.
     */
    bp::object
    py_get_noe_ndarray() const;

    /**
     * @brief Constructs the sum of weights ndarray for releasing it to
     *     Python.
     *     The lifetime of this new object and this ndhist object will be
     *     managed through the BoostNumpy ndarray_accessor_return() policy.
     * @note In case this ndhist is a zero-dimensional histogram this method
     *     returns a scalar.
     */
    bp::object
    py_get_sow_ndarray() const;

    /**
     * @brief Constructs the sum of weights squared ndarray for releasing
     *     it to Python.
     *     The lifetime of this new object and this ndhist object will be
     *     managed through the BoostNumpy ndarray_accessor_return() policy.
     * @note In case this ndhist is a zero-dimensional histogram this method
     *     returns a scalar.
     */
    bp::object
    py_get_sows_ndarray() const;

    /**
     * @brief Returns the ndarray holding the bin edges of the given axis.
     *        Note, that this is always a copy, since the edges are supposed
     *        to be readonly, because some axis types do not store the edges
     *        array internally.
     */
    bn::ndarray
    get_binedges_ndarray(intptr_t axis=0) const;

    /**
     * @brief In case of a 1-dimensional histogram it returns a ndarray holding
     *        the bin edges, otherwise a tuple of ndarray objects holding the
     *        bin edges for each axis.
     */
    bp::object
    py_get_binedges() const;

    /**
     * @brief Returns the ndarray holding the bin centers of the given axis.
     *        Note, that this is always a copy, since the edges are supposed
     *        to be readonly, because some axis types do not store the edges
     *        array internally.
     */
    bn::ndarray
    get_bincenters_ndarray(intptr_t axis=0) const;

    /**
     * @brief In case of a 1-dimensional histogram it returns a ndarray holding
     *        the bin center values, otherwise a tuple of ndarray objects
     *        holding the bin centers for each axis.
     */
    bp::object
    py_get_bincenters() const;

    /**
     * @brief Returns the ndarray holding the bin widths of the given axis.
     *        Note, that this is always a copy, since the edges are supposed
     *        to be readonly, because some axis types do not store the edges
     *        array internally.
     */
    bn::ndarray
    get_binwidths_ndarray(intptr_t axis=0) const;

    /**
     * @brief In case of a 1-dimensional histogram it returns a ndarray holding
     *        the bin width values, otherwise a tuple of ndarray objects
     *        holding the bin widths for each axis.
     */
    bp::object
    py_get_binwidths() const;

    /**
     * @brief Gets the title of the histogram.
     */
    std::string py_get_title() const
    {
        return title_;
    }

    /**
     * @brief Sets the title of the histogram to the given title string value.
     */
    void
    py_set_title(std::string const & title)
    {
        title_ = title;
    }

    /**
     * @brief Returns a tuple of str object holding the labels of each axis.
     */
    bp::tuple
    py_get_labels() const;

    /**
     * @brief Creates a tuple of length *nd* where each element is a
     *        *nd*-dimensional ndarray holding the underflow (number of entries)
     *        bins for the particular axis, where the index of the tuple element
     *        specifies the axis.
     *        The dimension of the particular axis is collapsed to
     *        one and the lengths of the other dimensions is extended by two.
     *        Each element of the returned ndarray is a copy of the histogram's
     *        bin.
     */
    bp::tuple
    py_get_underflow_entries() const;

    /**
     * @brief Same as py_get_underflow_entries() but instead each returned
     *     ndarray is an actual view into the histogram's bin content array.
     */
    bp::tuple
    py_get_underflow_entries_view() const;

    /**
     * @brief Same as ``py_get_underflow_entries`` but for the overflow (number
     *        of entries) bins.
     */
    bp::tuple
    py_get_overflow_entries() const;

    /**
     * @brief Same as ``py_get_underflow_entries_view`` but for the overflow
     *     (number of entries) bins.
     */
    bp::tuple
    py_get_overflow_entries_view() const;

    /**
     * @brief Creates a tuple of length *nd* where each element is a
     *        *nd*-dimensional ndarray holding the underflow (sum of weights)
     *        bins for the particular axis, where the index of the tuple element
     *        specifies the axis.
     *        The dimension of the particular axis is collapsed to
     *        one and the lengths of the other dimensions is extended by two.
     *        Each ndarray holds a copy of the histogram bins.
     *        Example: For (3,2) shaped two-dimensional histogram, there will
     *                 be two tuple elements with a two-dimensional ndarray
     *                 each. The shape of the first array (i.e. for the first
     *                 axis) will be (1,4) and the shape of the second array
     *                 will be (5,1).
     */
    bp::tuple
    py_get_underflow() const;

    /**
     * @brief See the documentation of py_get_underflow().
     *     But each returned ndarray is an actual view into the internal bin
     *     content array of the histogram.
     */
    bp::tuple
    py_get_underflow_view() const;

    /**
     * @brief Creates a tuple of length *nd* where each element is a
     *        *nd*-dimensional ndarray holding the overflow (sum of weights)
     *        bins for the
     *        particular axis, where the index of the tuple element specifies
     *        the axis. The dimension of the particular axis is collapsed to
     *        one and the lengths of the other dimensions is extended by two.
     */
    bp::tuple
    py_get_overflow() const;

    /**
     * @brief See the documentation of py_get_overflow().
     *     But each returned ndarray is an actual view into the internal bin
     *     content array of the histogram.
     */
    bp::tuple
    py_get_overflow_view() const;

    /**
     * @brief Creates a tuple of length *nd* where each element is a
     *        *nd*-dimensional ndarray holding the underflow (sum of weights
     *        squared) bins for the particular axis, where the index of the
     *        tuple element specifies
     *        the axis. The dimension of the particular axis is collapsed to
     *        one and the lengths of the other dimensions is extended by two.
     */
    bp::tuple
    py_get_underflow_squaredweights() const;

    /**
     * @brief Same as py_get_underflow_squaredweights() but the returned
     *     ndarrays are actual views into the bin content array of the
     *     histogram.
     */
    bp::tuple
    py_get_underflow_squaredweights_view() const;

    /**
     * @brief Creates a tuple of length *nd* where each element is a
     *     *nd*-dimensional ndarray holding the overflow (sum of weights
     *     squared) bins for the particular axis, where the index of the
     *     tuple element specifies
     *     the axis. The dimension of the particular axis is collapsed to
     *     one and the lengths of the other dimensions is extended by two.
     */
    bp::tuple
    py_get_overflow_squaredweights() const;

    /**
     * @brief Same as py_get_overflow_squaredweights() but the returned
     *     ndarrays are actual views into the bin content array of the
     *     histogram.
     */
    bp::tuple
    py_get_overflow_squaredweights_view() const;

    /**
     * @brief Fills a given n-dimension value into the histogram's bin content
     *     array.
     *     On the Python side, the *ndvalue* is a numpy object array that might
     *     hold values of different types. The order of these types must match
     *     the types of the bin edges of the axes.
     */
    void
    py_fill(bp::object const & ndvalue_obj, bp::object weight_obj);

    boost::shared_ptr<ndhist>
    py_get_base() const
    {
        return boost::const_pointer_cast<ndhist, ndhist const>(base_);
    }

    /**
     * @brief Create a new ndhist from this ndhist object, where only the
     *        specified dimensions are included and the others are summed over.
     */
    ndhist
    project(bp::object const & dims) const;

    inline
    std::vector< boost::shared_ptr<Axis> > &
    get_axes()
    {
        return axes_;
    }

    inline
    detail::ndarray_storage &
    get_bc_storage()
    {
        return bc_;
    }

    inline
    uintptr_t
    get_nd() const
    {
        return nd_;
    }

    inline
    bn::dtype
    get_ndvalues_dtype() const
    {
        return ndvalues_dt_;
    }

    inline
    bn::dtype
    get_weight_dtype() const
    {
        return bc_weight_dt_;
    }

    inline
    bp::object
    get_weight_class() const
    {
        return bc_class_;
    }

    /**
     * @brief Checks if this ndhist object shares the bin content array with an
     *     other ndhist object, i.e. does not own the data.
     */
    bool
    is_view() const
    {
        return (base_ != NULL);
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

    static
    void
    initialize_extended_array_axis(
        bp::object & arr_obj
      , bp::object const & obj_class
      , intptr_t axis
      , intptr_t f_n_extra_bins
      , intptr_t b_n_extra_bins
    );

    /**
     * @brief Constructs a ndarray object that is a view into the bin content
     *     bytearray that includes the under- and overflow bins for extendable
     *     axes.
     * @note The returned ndarray object has no base object set, but also does
     *     not have the OWNDATA flag set.
     */
    bn::ndarray
    construct_complete_bin_content_ndarray(
        bn::dtype const & dt
      , size_t const field_idx=0
    ) const;

    template <typename WeightValueType>
    detail::ValueCache<WeightValueType> &
    get_value_cache()
    {
        return *static_cast<detail::ValueCache<WeightValueType> *>(value_cache_.get());
    }

  private:
    ndhist()
      : nd_(0)
      , ndvalues_dt_(bn::dtype::new_builtin<void>())
      , bc_noe_dt_(bn::dtype::get_builtin<uintptr_t>())
      , bc_weight_dt_(bn::dtype::get_builtin<void>())
      , bc_class_(bp::object())
    {};

  protected:
    /**
     * @brief Setups the ndhist's function pointers based on the weight data
     *     type.
     */
    void
    setup_function_pointers();

    /**
     * @brief Setups the ndhist's value cache, that depends on the weight data
     *     type.
     */
    void
    setup_value_cache(intptr_t const value_cache_size);

    /**
     * @brief Calculates the shape, front and back capacities needed for a view
     *     into the bin content array that represents only the core bin content
     *     array, i.e. excluding the under- and overflow bins.
     */
    void
    calc_core_bin_content_ndarray_settings(
        std::vector<intptr_t> & shape
      , std::vector<intptr_t> & front_capacity
      , std::vector<intptr_t> & back_capacity
    ) const;

  public:
    /** The number of dimenions of this histogram.
     */
    uintptr_t const nd_;

    /** The dtype object describing the ndvalues structure. It describes a
     *  structured ndarray with field names, one for each axis of the histogram.
     */
    bn::dtype ndvalues_dt_;

    /** The list of pointers to the Axis object for each dimension.
     */
    std::vector< boost::shared_ptr<Axis> > axes_;

    std::vector<intptr_t> axes_extension_max_fcap_vec_;
    std::vector<intptr_t> axes_extension_max_bcap_vec_;

    /** The bin content.
     *  bc_ holds the actual data of the bins.
     *  bc_noe_dt_ is the dtype object describing the data type of the number
     *      of entries.
     *  bc_weight_dt_ is the dtype object describing the data type of the
     *      weights.
     */
    detail::ndarray_storage bc_;
    bn::dtype const bc_noe_dt_;
    bn::dtype const bc_weight_dt_;

    /** The Python object holding the class object to initialize the bin
     *  content elements if the datatype is object.
     */
    bp::object const bc_class_;

    boost::shared_ptr<detail::ValueCacheBase> value_cache_;

    boost::function<void (ndhist &, ndhist const &)> iadd_fct_;
    boost::function<void (ndhist &, bn::ndarray const &)> idiv_fct_;
    boost::function<void (ndhist &, bn::ndarray const &)> imul_fct_;
    boost::function<std::vector<bn::ndarray> (ndhist const &, axis::out_of_range_t const, size_t const)> get_noe_type_field_axes_oor_ndarrays_fct_;
    boost::function<std::vector<bn::ndarray> (ndhist const &, axis::out_of_range_t const, size_t const)> get_weight_type_field_axes_oor_ndarrays_fct_;
    boost::function<void (ndhist &, bp::object const &, bp::object const &)> fill_fct_;
    boost::function<ndhist (ndhist const &, std::set<intptr_t> const &)> project_fct_;

    /** The title string of the histogram, useful for plotting purposes.
     */
    std::string title_;

    /** The ndhist object owning the bin content array, in cases this ndhist
     *  object provides a data view into that ndhist object.
     */
    boost::shared_ptr<ndhist const> base_;
};

template <typename T>
ndhist &
ndhist::
operator*=(T const & rhs)
{
    // Create a bp::object from the rhs value and check if it is a scalar.
    bp::object value_obj(rhs);
    if(! bn::is_any_scalar(value_obj))
    {
        std::stringstream ss;
        ss << "The *= operator is only defined for scalar values!";
        throw ValueError(ss.str());
    }

    // Create a (scalar) ndarray object with a data type of the histogram's
    // bin content weights (performing automatic type conversion).
    bn::ndarray value_arr = bn::from_object(value_obj, bc_weight_dt_);
    imul_fct_(*this, value_arr);
    return *this;
}

template <typename T>
ndhist &
ndhist::
operator/=(T const & rhs)
{
    // Create a bp::object from the rhs value and check if it is a scalar.
    bp::object value_obj(rhs);
    if(! bn::is_any_scalar(value_obj))
    {
        std::stringstream ss;
        ss << "The /= operator is only defined for scalar values!";
        throw ValueError(ss.str());
    }

    // Create a (scalar) ndarray object with a data type of the histogram's
    // bin content weights (performing automatic type conversion).
    bn::ndarray value_arr = bn::from_object(value_obj, bc_weight_dt_);
    idiv_fct_(*this, value_arr);
    return *this;
}

template <typename T>
ndhist
ndhist::
operator*(T const & rhs) const
{
    ndhist newhist = this->empty_like();
    newhist += *this;
    newhist *= rhs;
    return newhist;
}

template <typename T>
ndhist
ndhist::
operator/(T const & rhs) const
{
    ndhist newhist = this->empty_like();
    newhist += *this;
    newhist /= rhs;
    return newhist;
}

}// namespace ndhist

#endif // !NDHIST_NDHIST_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define ND BOOST_PP_ITERATION()

template <>
struct specific_nd_traits<ND>
{
    template<typename BCValueType>
    struct fill_traits
    {
        static
        void
        fill(ndhist & self, bp::object const & ndvalues_obj, bp::object const & weight_obj)
        {
            //std::cout << "specific_nd_traits<"<< BOOST_PP_STRINGIZE(ND) <<">::fill_traits<BCValueType>::fill" << std::endl;
            if(! PyTuple_Check(ndvalues_obj.ptr()))
            {
                // The input ndvalues object is not a tuple, so we assume it's a
                // structured array, which will be handled by the
                // generic_nd_traits.
                //FIXME generic_nd_traits::fill_traits<BCValueType>::fill(self, ndvalues_obj, weight_obj);
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

            loop_service_t loop_service(BOOST_PP_ENUM_PARAMS(ND, ndvalue_arr_service), weight_arr_service);

            #define NDHIST_DEF(z, n, data) \
                bn::detail::iter_operand BOOST_PP_CAT(ndvalue_arr_iter_op,n)( BOOST_PP_CAT(ndvalue_arr_service,n).get_arr(), bn::detail::iter_operand::flags::READONLY::value, BOOST_PP_CAT(ndvalue_arr_service,n).get_arr_bcr_data() );
            BOOST_PP_REPEAT(ND, NDHIST_DEF, ~)
            #undef NDHIST_DEF
            bn::detail::iter_operand weight_arr_iter_op( weight_arr_service.get_arr(), bn::detail::iter_operand::flags::READONLY::value, weight_arr_service.get_arr_bcr_data() );

            bn::detail::iter_flags_t iter_flags =
                bn::detail::iter::flags::REFS_OK::value // This is needed for the
                                                        // weight, which can be bp::object.
              | bn::detail::iter::flags::EXTERNAL_LOOP::value
              | bn::detail::iter::flags::BUFFERED::value
              | bn::detail::iter::flags::GROWINNER::value;
            bn::order_t order = bn::KEEPORDER;
            bn::casting_t casting = bn::NO_CASTING;
            intptr_t buffersize = 0; // Use the default value.

            bn::detail::iter iter(
                  iter_flags
                , order
                , casting
                , loop_service.get_loop_nd()
                , loop_service.get_loop_shape_data()
                , buffersize
                , BOOST_PP_ENUM_PARAMS(ND, ndvalue_arr_iter_op)
                , weight_arr_iter_op
            );
            iter.init_full_iteration();

            // Get a handle on the value cache.
            detail::ValueCache<BCValueType> & value_cache = self.get_value_cache<BCValueType>();

            // Do the iteration.
            std::vector<intptr_t> indices(ND, 0);
            std::vector<intptr_t> relative_indices(ND, 0);
            std::vector<intptr_t> f_n_extra_bins_vec(ND, 0);
            std::vector<intptr_t> b_n_extra_bins_vec(ND, 0);
            std::vector<intptr_t> const & bc_fcap = self.bc_.get_front_capacity_vector();
            std::vector<intptr_t> const & bc_bcap = self.bc_.get_back_capacity_vector();
            bool is_oor;
            bool extend_axes;
            bool reallocation_upon_extension = false;
            bool value_cached;
            ::ndhist::axis::out_of_range_t oor_flag;
            intptr_t bc_data_offset = self.bc_.get_bytearray_data_offset() + self.bc_.calc_first_shape_element_data_offset();
            char * bc_data_addr;
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Get the weight scalar from the iterator.
                    typename bin_utils<BCValueType>::weight_ref_type weight = bin_utils<BCValueType>::get_weight_type_value_from_iter(iter, ND);

                    // Fill the scalar ndvalue into the bin content array.
                    // Get the coordinate of the current ndvalue.
                    is_oor = false;
                    extend_axes = false;
                    value_cached = false;

                    std::vector<intptr_t> const & bc_data_strides = self.bc_.get_data_strides_vector();
                    bc_data_addr = self.bc_.get_data() + bc_data_offset;
                    for(size_t i=0; i<ND; ++i)
                    {
                        // Don't waste time for values, which can't be filled
                        // anyways.
                        if(is_oor) break;

                        //std::cout << "tuple fill: Get bin idx of axis " << i << " of " << ND << std::endl;
                        Axis & axis = *self.axes_[i];
                        char * const ndvalue_ptr = iter.data_ptr_array_ptr_[i];
                        intptr_t const bin_idx = axis.get_bin_index(ndvalue_ptr, oor_flag);
                        std::cout << "bin_idx = "<<bin_idx<<std::endl;
                        if(oor_flag == ::ndhist::axis::OOR_NONE)
                        {
                            // The current value fits into the current axis
                            // range.
                            //std::cout << "normal fill i=" << i << "indices.size()="<<indices.size();
                            //std::cout << "relative_indices.size() "<< relative_indices.size()<<std::endl;
                            bc_data_addr += bin_idx*bc_data_strides[i];

                            indices[i] = bin_idx;
                            relative_indices[i] = bin_idx;
                        }
                        else
                        {
                            // The current value does not fit into the current
                            // axis range. But the axis might be extendable.
                            if(axis.is_extendable())
                            {
                                //std::cout << "axis is extentable" << std::endl;
                                intptr_t const n_extra_bins = axis.request_extension(ndvalue_ptr, oor_flag);
                                if(oor_flag == ::ndhist::axis::OOR_UNDERFLOW)
                                {
                                    indices[i] = 0;
                                    relative_indices[i] = n_extra_bins;

                                    f_n_extra_bins_vec[i] = std::max(-n_extra_bins, f_n_extra_bins_vec[i]);
                                    reallocation_upon_extension |= (f_n_extra_bins_vec[i] > bc_fcap[i]);
                                }
                                else // oor_flag == ::ndhist::axis::OOR_OVERFLOW
                                {
                                    intptr_t const index = axis.get_n_bins() + n_extra_bins - 1;

                                    indices[i] = index;
                                    relative_indices[i] = index;

                                    b_n_extra_bins_vec[i] = std::max(n_extra_bins, b_n_extra_bins_vec[i]);
                                    reallocation_upon_extension |= (b_n_extra_bins_vec[i] > bc_bcap[i]);
                                }

                                extend_axes = true;
                            }
                            else
                            {
                                //std::cout << "axis is NOT extendable" << std::endl;
                                // The current value is out-of-range on the
                                // current axis, which is not extendable.
                                // So mark this ndvalue as oor.
                                is_oor = true;
                            }
                        }
                    }

                    if(is_oor)
                    {
                        // There is at least one axis, where the value is
                        // out-of-range, so there is no way to fill this
                        // ndvalue. So just skip it.
                        continue;
                    }

                    // If the value can be filled but an axis needs to get
                    // extended in order to do so, we want to cache the value if
                    // the extension would trigger a reallocation of memory.
                    if(extend_axes)
                    {
                        //std::cout << "extend_axes is true, size="<< oorfrstack.get_size() << std::endl<<std::flush;
                        // Check if an actual reallocation is required,
                        // if not, just extend the axes and fill it. Otherwise,
                        // fill the value into the value cache.
                        if(reallocation_upon_extension)
                        {
                            //std::cout << "reallocation required upon extension " << std::endl<<std::flush;
                            // Push the value into the value cache stack.
                            // If it returns ``true`` the cache is full and we
                            // need to extent the axes and fill the cached
                            // values in.
                            value_cached = true;
                            if(value_cache.push_back(relative_indices, weight))
                            {
                                //std::cout << "The stack is full. Flush it." << std::endl<<std::flush;
                                self.extend_axes(f_n_extra_bins_vec, b_n_extra_bins_vec);
                                self.extend_bin_content_array(f_n_extra_bins_vec, b_n_extra_bins_vec);
                                bc_data_offset = self.bc_.get_bytearray_data_offset() + self.bc_.calc_first_shape_element_data_offset();

                                flush_value_cache<BCValueType>(self, value_cache, f_n_extra_bins_vec, bc_data_offset);

                                memset(&f_n_extra_bins_vec.front(), 0, ND*sizeof(intptr_t));
                                memset(&b_n_extra_bins_vec.front(), 0, ND*sizeof(intptr_t));
                                reallocation_upon_extension = false;
                            }
                        }
                        else
                        {
                            // No reallocation of memory is required for the
                            // extension, so we just extend the axes.
                            //std::cout << "no reallocation required upon extension " << std::endl<<std::flush;
                            self.extend_axes(f_n_extra_bins_vec, b_n_extra_bins_vec);
                            self.extend_bin_content_array(f_n_extra_bins_vec, b_n_extra_bins_vec);
                            bc_data_offset = self.bc_.get_bytearray_data_offset() + self.bc_.calc_first_shape_element_data_offset();
                            memset(&f_n_extra_bins_vec.front(), 0, ND*sizeof(intptr_t));
                            memset(&b_n_extra_bins_vec.front(), 0, ND*sizeof(intptr_t));

                            // Since the strides have changed, we need to
                            // recompute the bc_data_addr.
                            std::vector<intptr_t> const & bc_data_strides = self.bc_.get_data_strides_vector();
                            bc_data_addr = self.bc_.get_data() + bc_data_offset;
                            for(size_t i=0; i<ND; ++i)
                            {
                                bc_data_addr += indices[i]*bc_data_strides[i];
                            }
                        }
                    }

                    if(! value_cached)
                    {
                        //std::cout << "increment the bin "<< std::endl<<std::flush;
                        detail::bin_utils<BCValueType>::increment_bin(bc_data_addr, weight);
                    }

                    // Jump to the next fill iteration.
                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());

            // Fill the remaining cached values.
            if(value_cache.get_size() > 0)
            {
                self.extend_axes(f_n_extra_bins_vec, b_n_extra_bins_vec);
                self.extend_bin_content_array(f_n_extra_bins_vec, b_n_extra_bins_vec);
                bc_data_offset = self.bc_.get_bytearray_data_offset() + self.bc_.calc_first_shape_element_data_offset();

                flush_value_cache<BCValueType>(self, value_cache, f_n_extra_bins_vec, bc_data_offset);
            }
        }
    };
};

#undef ND

#else
#if BOOST_PP_ITERATION_FLAGS() == 2

#define ND BOOST_PP_ITERATION()

#define NDHIST_SPECIFIC_ND_TRAITS_WEIGHT_VALUE_TYPE_SUPPORT(r, data, WEIGHT_VALUE_TYPE)  \
    if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<WEIGHT_VALUE_TYPE>()))\
    {                                                                          \
        if(bc_dtype_supported)                                                 \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "The bin content data type is supported by more than one "   \
               << "possible C++ data type! This is an internal error!";        \
            throw TypeError(ss.str());                                         \
        }                                                                      \
        fill_fct_ = &detail::specific_nd_traits<ND>::fill_traits<WEIGHT_VALUE_TYPE>::fill;\
        bc_dtype_supported = true;                                             \
    }

#if ND > 1
else
#endif
if(nd_ == ND)
{
    BOOST_PP_SEQ_FOR_EACH(NDHIST_SPECIFIC_ND_TRAITS_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)
}

#undef NDHIST_SPECIFIC_ND_TRAITS_WEIGHT_VALUE_TYPE_SUPPORT

#undef ND

#endif // BOOST_PP_ITERATION_FLAGS() == 2
#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
