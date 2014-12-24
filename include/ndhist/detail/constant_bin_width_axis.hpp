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

#include <ndhist/detail/axis.hpp>

namespace ndhist {
namespace detail {

template<typename AxisValueType>
struct ConstantBinWidthAxisData
  : AxisData
{
    intptr_t      n_bins_;
    AxisValueType bin_width_;
    AxisValueType min_;
};

template <typename AxisValueType>
struct ConstantBinWidthAxis
  : Axis
{
    typedef AxisValueType
            axis_value_type;
    typedef ConstantBinWidthAxisData<AxisValueType>
            axis_data_type;

    ConstantBinWidthAxis(
        bn::ndarray const & edges
      , intptr_t extension_max_fcap
      , intptr_t extension_max_bcap
    )
      : Axis(edges.get_dtype(), extension_max_fcap, extension_max_bcap)
    {
        // Set up the axis's function pointers.
        get_bin_index_fct     = &get_bin_index;
        request_extension_fct = &request_extension;
        extend_fct            = &extend;
        get_edges_ndarray_fct = &get_edges_ndarray;
        get_n_bins_fct        = &get_n_bins;

        data_ = boost::shared_ptr< axis_data_type >(new axis_data_type());
        axis_data_type & ddata = *static_cast<axis_data_type*>(data_.get());
        ddata.n_bins_ = edges.shape(0) - 1;
        bn::flat_iterator<axis_value_type> edges_iter(edges);
        bn::flat_iterator<axis_value_type> edges_iter_end(edges_iter.end());

        ddata.min_ = *edges_iter;
        ++edges_iter;
        axis_value_type const value = *edges_iter;
        ddata.bin_width_ = value - ddata.min_;
    }

    static
    intptr_t
    get_n_bins(boost::shared_ptr<AxisData> & axisdata)
    {
        axis_data_type & data = *static_cast<axis_data_type*>(axisdata.get());
        return data.n_bins_;
    }

    static
    bn::ndarray
    get_edges_ndarray(boost::shared_ptr<AxisData> & axisdata)
    {
        axis_data_type & data = *static_cast<axis_data_type*>(axisdata.get());
        intptr_t shape[1];
        shape[0] = data.n_bins_ + 1;
        bn::ndarray edges = bn::empty(1, shape, bn::dtype::get_builtin<axis_value_type>());
        bn::flat_iterator<axis_value_type> iter(edges);
        bn::flat_iterator<axis_value_type> iter_end(iter.end());
        for(intptr_t idx = 0; iter != iter_end; ++idx, ++iter)
        {
            axis_value_type value = *iter;
            value = data.min_ + idx*data.bin_width_;
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
