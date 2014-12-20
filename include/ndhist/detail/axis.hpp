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
#ifndef NDHIST_DETAIL_AXIS_HPP_INCLUDED
#define NDHIST_DETAIL_AXIS_HPP_INCLUDED 1

#include <boost/function.hpp>
#include <boost/python.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/detail/ndarray_storage.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace detail {

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

struct AxisData;

struct Axis
{
    Axis()
      : dt_(bn::dtype::get_builtin<void>())
      , autoscale_fcap_(0)
      , autoscale_bcap_(0)
    {}

    Axis(
        bn::dtype const & dt
      , intptr_t autoscale_fcap=0
      , intptr_t autoscale_bcap=0
    )
      : dt_(dt)
      , autoscale_fcap_(autoscale_fcap)
      , autoscale_bcap_(autoscale_bcap)
    {}

    bn::dtype & get_dtype() { return dt_; }

    bool has_autoscale() const
    {
        std::cout << " autoscale_fcap_ = "<< autoscale_fcap_
                  << " autoscale_bcap_ = "<< autoscale_bcap_<<std::endl;
        return ((autoscale_fcap_ > 0 && autoscale_bcap_ >= 0) ||
                (autoscale_bcap_ > 0 && autoscale_fcap_ >= 0));
    }

    bn::dtype dt_;
    intptr_t autoscale_fcap_;
    intptr_t autoscale_bcap_;
    boost::function<intptr_t (boost::shared_ptr<AxisData>, char *, axis::out_of_range_t *)> get_bin_index_fct;
    boost::function<intptr_t (boost::shared_ptr<AxisData>, char *, axis::out_of_range_t)> autoscale_fct;
    boost::function<intptr_t (boost::shared_ptr<AxisData>)> get_n_bins_fct;
    boost::function<bn::ndarray (boost::shared_ptr<AxisData>)> get_edges_ndarray_fct;
    boost::shared_ptr<AxisData> data_;
};

struct AxisData
{};

}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_AXIS_HPP_INCLUDED
