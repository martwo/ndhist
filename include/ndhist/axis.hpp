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

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/detail/axis.hpp>

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
      , is_extendable_(false)
      , extension_max_fcap_(0)
      , extension_max_bcap_(0)
    {}

    Axis(
        boost::numpy::dtype const & dt
      , std::string const & label
      , bool is_extendable=false
      , intptr_t extension_max_fcap=0
      , intptr_t extension_max_bcap=0
    )
      : dt_(dt)
      , label_(label)
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

}// namespace ndhist

#endif // NDHIST_AXIS_HPP_INCLUDED
