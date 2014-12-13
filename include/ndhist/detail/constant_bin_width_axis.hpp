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

#include <ndhist/detail/axis.hpp>

namespace ndhist {
namespace detail {

template<typename AxisValueType>
struct ConstantBinWidthAxisData : AxisData
{
    intptr_t      n_bins_;
    AxisValueType bin_width_;
    AxisValueType min_;
};

template <class Derived>
struct ConstantBinWidthAxisBase
  : Axis
{
    ConstantBinWidthAxisBase(
        bn::ndarray const & edges
    )
    {
        typedef typename Derived::axis_value_type
                axis_value_type;

        // Set up the axis's function pointers.
        get_bin_index_fct     = &Derived::get_bin_index;
        get_edges_ndarray_fct = &Derived::get_edges_ndarray;

        data_ = boost::shared_ptr< typename Derived::axis_data_type >(new (typename Derived::axis_data_type)());
        typename Derived::axis_data_type & ddata = *static_cast<typename Derived::axis_data_type*>(data_.get());
        ddata.n_bins_ = edges.shape(0) - 1;
        bn::flat_iterator<axis_value_type> edges_iter(edges);
        bn::flat_iterator<axis_value_type> edges_iter_end;
        ddata.min_ = *edges_iter;
        ++edges_iter;
        ddata.bin_width_ = *edges_iter - ddata.min_;
        //std::cout << "ddata.bin_width_ = " << ddata.bin_width_ << std::endl;
    }

    static
    bn::ndarray
    get_edges_ndarray(boost::shared_ptr<AxisData> axisdata)
    {
        typedef typename Derived::axis_value_type
                axis_value_type;

        typename Derived::axis_data_type & data = *static_cast<typename Derived::axis_data_type*>(axisdata.get());
        intptr_t shape[1];
        shape[0] = data.n_bins_ + 1;
        bn::ndarray edges = bn::empty(1, shape, bn::dtype::get_builtin<axis_value_type>());
        bn::flat_iterator<axis_value_type> iter(edges);
        for(intptr_t idx = 0; iter != iter.end; ++idx, ++iter)
        {
            axis_value_type & value = *iter;
            value = data.min_ + idx*data.bin_width_;
        }
        return edges;
    }
};

template <typename AxisValueType>
struct ConstantBinWidthAxis
  : ConstantBinWidthAxisBase< ConstantBinWidthAxis<AxisValueType> >
{
    typedef ConstantBinWidthAxisBase< ConstantBinWidthAxis<AxisValueType> >
            base_t;
    typedef AxisValueType axis_value_type;
    typedef ConstantBinWidthAxisData<AxisValueType> axis_data_type;

    ConstantBinWidthAxis(bn::ndarray const & edges)
      : base_t(edges)
    {}

    static
    intptr_t
    get_bin_index(boost::shared_ptr<AxisData> axisdata, char * value_ptr)
    {
        axis_data_type & data = *static_cast< axis_data_type *>(axisdata.get());
        AxisValueType const & value = *reinterpret_cast<AxisValueType*>(value_ptr);
        if(value - data.min_ < 0)
        {
            //std::cout << "underflow: " << value << ", min = "<< data.min_ << std::endl;
            return -1;
        }
        intptr_t const idx = (value - data.min_)/data.bin_width_;
        if(idx >= data.n_bins_)
        {
            //std::cout << "overflow: " << value << ", idx = "<< idx << std::endl;
            return -2;
        }
        //std::cout << "value " << value << " at " << idx << std::endl;
        return idx;
    }
};

//FIXME: Have specialization for bool and bp::object.

}//namespace detail
}//namespace ndhist

#endif // NDHIST_DETAIL_CONSTANT_BIN_WIDTH_AXIS_HPP_INCLUDED
