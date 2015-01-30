/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <ndhist@martin-wolf.org>
 *
 * This file is distributed under the BSD 2-Clause Open Source License
 * (See LICENSE file).
 *
 */
#ifndef NDHIST_AXIS_HPP_INCLUDED
#define NDHIST_AXIS_HPP_INCLUDED 1

#include <string>
#include <sstream>

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/detail/axis.hpp>
#include <ndhist/type_support.hpp>

namespace ndhist {

namespace axis {

/** The enum type for describing the type of an out of range event.
 */
enum out_of_range_t
{
    OOR_NONE      =  0,
    OOR_UNDERFLOW = -1,
    OOR_OVERFLOW  = -2
};

}// namespace axis

class Axis
{
  public:
    Axis()
      : dt_(boost::numpy::dtype::get_builtin<void>())
      , label_(std::string(""))
      , name_(std::string(""))
      , is_extendable_(false)
      , extension_max_fcap_(0)
      , extension_max_bcap_(0)
    {}

    Axis(
        boost::numpy::dtype const & dt
      , std::string const & label
      , std::string const & name
      , bool is_extendable=false
      , intptr_t extension_max_fcap=0
      , intptr_t extension_max_bcap=0
    )
      : dt_(dt)
      , label_(label)
      , name_(name)
      , is_extendable_(is_extendable)
      , extension_max_fcap_(extension_max_fcap)
      , extension_max_bcap_(extension_max_bcap)
    {}

    /** Returns a reference to the dtype object of the axis values.
     */
    boost::numpy::dtype const &
    get_dtype() const
    {
        return dt_;
    }

    /** Returns a reference to the name std::string object of this axis.
     */
    std::string &
    get_name()
    {
        return name_;
    }

    void
    set_name(std::string const & name)
    {
        name_ = name;
    }

    /** Returns ``true`` if the axis is extendable.
     */
    bool
    is_extendable() const
    {
        return is_extendable_;
    }

    /** Wraps the given other Axis object. This means, it copies the member
     *  values from the given Axis object to this Axis object. Also the function
     *  pointers!
     *  Note: This Axis object must own the given other Axis object, so the
     *        pointers don't become invalid!
     */
    void
    wrap_axis(Axis const & axis)
    {
        dt_                    = axis.dt_;
        label_                 = axis.label_;
        name_                  = axis.name_;
        is_extendable_         = axis.is_extendable_;
        extension_max_fcap_    = axis.extension_max_fcap_;
        extension_max_bcap_    = axis.extension_max_bcap_;

        get_bin_index_fct_     = axis.get_bin_index_fct_;
        get_edges_ndarray_fct_ = axis.get_edges_ndarray_fct_;
        get_n_bins_fct_        = axis.get_n_bins_fct_;
        request_extension_fct_ = axis.request_extension_fct_;
        extend_fct_            = axis.extend_fct_;
    }

    /** The data type of the axis values.
     */
    boost::numpy::dtype dt_;

    /** The label of the axis.
     */
    std::string label_;

    /** The name of the axis. This name is used to name the values of this axis
     *  inside a structured numpy ndarray.
     */
    std::string name_;

    /** Flag if the axis is extendable (true) or not (false).
     */
    bool is_extendable_;

    /** The maximum front capacity (number of extra bins at the beginning of
     *  the axis) in case the axis is extendable.
     */
    intptr_t extension_max_fcap_;

    /** The maximum back capacity (number of extra bins at the end of the axis)
     *  in case the axis is extendable.
     */
    intptr_t extension_max_bcap_;

    /** This function is supposed to get the axis's bin index for the given data
     *  value (which is stored in memory at the given address).
     *  In case the value lies outside of the axis range
     *  (including the possible under- and overflow bins of the axis), the
     *  out_of_range variable must be set accordingly. In that case the return
     *  value of this function is undefined.
     */
    boost::function<intptr_t (boost::shared_ptr<Axis> &, char *, axis::out_of_range_t &)>
        get_bin_index_fct_;

    /** This function is supposed to return (a copy of) the edges array
     *  (including the possible under- and overflow bins) as a
     *  boost::numpy::ndarray object.
     */
    boost::function<boost::numpy::ndarray (boost::shared_ptr<Axis> &)>
        get_edges_ndarray_fct_;

    /** This function is supposed to return the number of bins of the axis
     *  (including the possible under- and overflow bins).
     */
    boost::function<intptr_t (boost::shared_ptr<Axis> &)>
        get_n_bins_fct_;

    /** This function is supposed to calculate the number of bins, that would
     *  have to be added to the left (negative returned value) or to the right
     *  (positive returned value) of the axis, in order to be able to contain
     *  the value (which is stored in memory at the given address) on the axis.
     *  The out_of_range constant provides a hint in what direction the axis
     *  needs to get extended.
     *  Note: This function is only called, when the axis is extendable, thus
     *        there are no under- and overflow bins defined in those cases.
     */
    boost::function<intptr_t (boost::shared_ptr<Axis> &, char *, axis::out_of_range_t const)>
        request_extension_fct_;

    /** This function is supposed to extend the axis by the given number of bins
     *  to the left and the right of the axis, respectively.
     *  Note: This function is only called, when the axis is extendable, thus
     *        there are no under- and overflow bins defined in those cases.
     */
    boost::function<void (boost::shared_ptr<Axis> &, intptr_t, intptr_t)>
        extend_fct_;
};

namespace detail {

/** The PyAxisWrapper template provides a wrapper for an axis template that
 *  depends on the axis value type.
 *  In order to expose that axis template to Python, we need to get rid of the
 *  axis value type template parameter. The PyAxisWrapper will do that.
 *
 *  The requirement on the AxisTypeTemplate is that it has a member type named
 *  ``type`` which is the type of the to-be-wrapped axis object class.
 */
template <template <typename AxisValueType> AxisTypeTemplate>
class PyAxisWrapper
  : public Axis
{
  protected:
    // This is the actual axis object which is axis value type dependent.
    boost::shared_ptr<Axis> axis_;

  public:
    PyAxisWrapper(
        bn::ndarray const & edges
      , std::string const & label=std::string("")
      , std::string const & name=std::string("")
      , bool is_extendable=false
      , intptr_t extension_max_fcap=0
      , intptr_t extension_max_bcap=0
    )
    {
        // Create the AxisTypeTemplate<AxisValueType> object on the heap
        // and save a pointer to it. Then wrap this Axis object around the
        // created axis object. So we don't have to do type lookups whenever
        // calling an API function.
        bool axis_dtype_is_supported = false;
        bn::dtype axis_dtype = edges.get_dtype();
        #define NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT(r, data, AXIS_VALUE_TYPE)           \
            if(bn::dtype::equivalent(axis_dtype, bn::dtype::get_builtin<AXIS_VALUE_TYPE>()))\
            {                                                                         \
                if(axis_dtype_is_supported)                                           \
                {                                                                     \
                    std::stringstream ss;                                             \
                    ss << "The axis value data type is supported by more than one "   \
                       << "possible C++ data type! This is an internal error!";       \
                    throw TypeError(ss.str());                                        \
                }                                                                     \
                axis_ = boost::shared_ptr< typename AxisTypeTemplate<AXIS_VALUE_TYPE>::type >(new typename AxisTypeTemplate<AXIS_VALUE_TYPE>::type(edges, label, name, is_extendable, extension_max_fcap, axis_extension_max_bcap));\
                axis_dtype_is_supported = true;                                       \
            }
        BOOST_PP_SEQ_FOR_EACH(NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_AXIS_VALUE_TYPES)
        #undef NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT

        // Wrap the axis_ Axis object.
        wrap_axis(*axis_);
    }
};

}//namespace detail

}//namespace ndhist

#endif // NDHIST_AXIS_HPP_INCLUDED
