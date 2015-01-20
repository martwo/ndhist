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
#ifndef NDHIST_DETAIL_CONSTANT_BIN_WIDTH_AXIS_HPP_INCLUDED
#define NDHIST_DETAIL_CONSTANT_BIN_WIDTH_AXIS_HPP_INCLUDED 1

#include <cmath>
#include <string>

#include <boost/python/slice.hpp>

#include <boost/numpy/iterators/flat_iterator.hpp>

#include <ndhist/detail/axis.hpp>

namespace ndhist {
namespace detail {

template <typename AxisValueType>
struct ConstantBinWidthAxis
  : Axis
{
    typedef AxisValueType
            axis_value_type;
    typedef ConstantBinWidthAxis<AxisValueType>
            axis_type;

    intptr_t        n_bins_;
    axis_value_type bin_width_;
    axis_value_type min_;
    axis_value_type underflow_edge_;
    axis_value_type overflow_edge_;

    ConstantBinWidthAxis(
        bn::ndarray const & edges
      , std::string const & label
      , intptr_t extension_max_fcap
      , intptr_t extension_max_bcap
    )
      : Axis(edges.get_dtype(), label, extension_max_fcap, extension_max_bcap)
    {
        // Set up the axis's function pointers.
        get_bin_index_fct        = &get_bin_index;
        request_extension_fct    = &request_extension;
        extend_fct               = &extend;
        get_edges_ndarray_fct    = &get_edges_ndarray;
        get_n_bins_fct           = &get_n_bins;

        if(is_extendable())
        {
            n_bins_ = edges.shape(0) - 1;
        }
        else
        {
            n_bins_ = edges.shape(0) - 3;
        }
        if(n_bins_ <= 0)
        {
            std::stringstream ss;
            ss << "The edges array does not contain enough edges! Remember "
               << "that the edges of the out-or-range bins, need to be "
               << "specified as well, when the axis is not extendable.";
            throw ValueError(ss.str());
        }
        bn::iterators::flat_iterator< bn::iterators::single_value<axis_value_type> > edges_iter(edges);
        if(! is_extendable())
        {
            // Set and skip the underflow edge.
            underflow_edge_ = *edges_iter;
            ++edges_iter;
        }
        min_ = *edges_iter;
        ++edges_iter;
        axis_value_type const value = *edges_iter;
        bin_width_ = value - min_;
        if(! is_extendable())
        {
            // Set the overflow edge.
            overflow_edge_ = *(edges_iter.advance(edges_iter.distance_to(edges_iter.end()) - 1));
        }
    }

    static
    intptr_t
    get_n_bins(boost::shared_ptr<Axis> & axisptr)
    {
        axis_type & axis = *static_cast<axis_type*>(axisptr.get());
        return axis.n_bins_;
    }

    static
    bn::ndarray
    get_edges_ndarray(boost::shared_ptr<Axis> & axisptr)
    {
        axis_type const & axis = *static_cast<axis_type const *>(axisptr.get());
        intptr_t shape[1];
        if(axis.is_extendable()) {
            shape[0] = axis.n_bins_ + 1;
        }
        else {
            shape[0] = axis.n_bins_ + 3;
        }
        bn::ndarray edges = bn::empty(1, shape, bn::dtype::get_builtin<axis_value_type>());
        bn::iterators::flat_iterator< bn::iterators::single_value<axis_value_type> > iter(edges);
        if(! axis.is_extendable())
        {
            // Set the underflow edge.
            axis_value_type & value = *iter;
            value = axis.underflow_edge_;
            ++iter;
        }
        intptr_t idx = 0;
        while(! iter.is_end())
        {
            axis_value_type & value = *iter;
            value = axis.min_ + idx*axis.bin_width_;

            ++idx;
            ++iter;
        }
        if(! axis.is_extendable())
        {
            // Set the overflow edge.
            iter.advance(-1);
            axis_value_type & value = *iter;
            value = axis.overflow_edge_;
        }

        return edges;
    }

    static
    intptr_t
    get_bin_index(boost::shared_ptr<AxisData> & axisdata, char * value_ptr, axis::out_of_range_t * oor_ptr)
    {
        axis_data_type & data = *static_cast< axis_data_type *>(axisdata.get());
        axis_value_type const & value = *reinterpret_cast<axis_value_type*>(value_ptr);

        //std::cout << "Got value: " << value << std::endl;
        if(value - data.min_ < 0)
        {
            //std::cout << "underflow: " << value << ", min = "<< data.min_ << std::endl;
            *oor_ptr = axis::OOR_UNDERFLOW;
            return -1;
        }
        intptr_t const idx = (value - data.min_)/data.bin_width_;
        if(idx >= data.n_bins_)
        {
            //std::cout << "overflow: " << value << ", idx = "<< idx << std::endl;
            *oor_ptr = axis::OOR_OVERFLOW;
            return -1;
        }
        //std::cout << "value " << value << " at " << idx << std::endl;
        *oor_ptr = axis::OOR_NONE;
        return idx;
    }

    // Determines the number of extra bins needed to the left (negative number
    // returned) or to the right (positive number returned) of the current axis
    // range.
    static
    intptr_t
    request_extension(boost::shared_ptr<AxisData> & axisdata, char * value_ptr, axis::out_of_range_t oor)
    {
        axis_data_type & data = *static_cast< axis_data_type *>(axisdata.get());
        axis_value_type const value = *reinterpret_cast<axis_value_type*>(value_ptr);

        if(oor == axis::OOR_UNDERFLOW)
        {
            intptr_t const n_extra_bins = std::ceil((std::abs(value - data.min_) / data.bin_width_));
            //std::cout << "request_autoscale (underflow): " << n_extra_bins << " extra bins." << std::endl<< std::flush;
            return -n_extra_bins;
        }
        else if(oor == axis::OOR_OVERFLOW)
        {
            intptr_t const n_extra_bins = intptr_t((value - data.min_)/data.bin_width_) - (data.n_bins_-1);
            //std::cout << "request_autoscale (overflow): " << n_extra_bins << " extra bins." << std::endl<< std::flush;
            return n_extra_bins;
        }

        return 0;
    }

    // Autoscales the axis range so that the given value just fits into the
    // new range. The returned integer values specifies how many bins have been
    // added to the left or to the right of the range.
    static
    void
    extend(boost::shared_ptr<AxisData> & axisdata, intptr_t f_n_extra_bins, intptr_t b_n_extra_bins)
    {
        axis_data_type & data = *static_cast< axis_data_type *>(axisdata.get());

        if(f_n_extra_bins > 0)
        {
            data.n_bins_ += f_n_extra_bins;
            data.min_    -= f_n_extra_bins * data.bin_width_;
        }
        if(b_n_extra_bins > 0)
        {
            data.n_bins_ += b_n_extra_bins;
        }

        //std::cout << "extend: " << f_n_extra_bins << " extra front bins, "<<b_n_extra_bins<<" extra back bins." << std::endl<< std::flush;
        //std::cout << "    new n_bins_ = "<< data.n_bins_ << std::endl<< std::flush;
        //std::cout << "    new min_ = "<< data.min_ << std::endl<< std::flush;
    }
};

}//namespace detail
}//namespace ndhist

#endif // NDHIST_DETAIL_CONSTANT_BIN_WIDTH_AXIS_HPP_INCLUDED
