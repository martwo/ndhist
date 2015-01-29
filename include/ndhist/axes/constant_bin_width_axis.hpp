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
#ifndef NDHIST_AXES_CONSTANT_BIN_WIDTH_AXIS_HPP_INCLUDED
#define NDHIST_AXES_CONSTANT_BIN_WIDTH_AXIS_HPP_INCLUDED 1

#include <cmath>
#include <string>
#include <sstream>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/iterators/flat_iterator.hpp>

#include <ndhist/type_support.hpp>
#include <ndhist/axis.hpp>
#include <ndhist/axes/generic_axis.hpp>
#include <ndhist/error.hpp>

namespace bn = boost::numpy;

namespace ndhist {
namespace axes {

template <typename AxisValueType>
class ConstantBinWidthAxis
  : public Axis
{
  public:
    typedef AxisValueType
            axis_value_type;

    typedef bn::iterators::single_value<axis_value_type>
            axis_value_type_traits;

    typedef constant_bin_width_axis<axis_value_type>
            axis_type;
    typedef axis_type
            type;

    /// The number of bins, including the possible under- and overflow bins.
    intptr_t n_bins_;

    /// The constant width of the bins. Remember, that the widths of the
    /// possible under- and overflow bins could be different from this value.
    axis_value_type bin_width_;

    /// The lower edge of the first bin with constant bin width, i.e. excluding
    /// the underflow bin.
    axis_value_type min_;

    /// The lower edge of the underflow bin.
    axis_value_type underflow_edge_;

    /// The upper edge of the overflow bin.
    axis_value_type overflow_edge_;

    ConstantBinWidthAxis(
        bn::ndarray const & edges
      , std::string const & label
      , bool is_extendable=false
      , intptr_t extension_max_fcap=0
      , intptr_t extension_max_bcap=0
    )
      : Axis(
            edges.get_dtype()
          , label
          , is_extendable
          , extension_max_fcap
          , extension_max_bcap
        )
    {
        // Set up the axis's function pointers.
        get_bin_index_fct_     = &get_bin_index;
        get_edges_ndarray_fct_ = &get_edges_ndarray;
        get_n_bins_fct_        = &get_n_bins;
        request_extension_fct_ = &request_extension;
        extend_fct_            = &extend;

        n_bins_ = edges.shape(0) - 1;
        if(n_bins_ <= 0)
        {
            std::stringstream ss;
            ss << "The edges array does not contain enough edges! Remember "
               << "that the edges of the out-or-range bins, need to be "
               << "specified as well, when the axis is not extendable.";
            throw ValueError(ss.str());
        }

        bn::iterators::flat_iterator< axis_value_type_traits > edges_iter(edges);

        // Set and skip the underflow edge.
        if(! is_extendable())
        {
            underflow_edge_ = *edges_iter;
            ++edges_iter;
        }

        min_ = *edges_iter;
        ++edges_iter;
        axis_value_type const value = *edges_iter;
        bin_width_ = value - min_;

        // Set the overflow edge.
        if(! is_extendable())
        {
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
        shape[0] = axis.n_bins_ + 1;
        bn::ndarray edges_arr = bn::empty(1, shape, bn::dtype::get_builtin<axis_value_type>());
        bn::iterators::flat_iterator< axis_value_type_traits > iter(edges_arr);
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

        return edges_arr;
    }

    static
    intptr_t
    get_bin_index(boost::shared_ptr<Axis> & axisptr, char * value_ptr, axis::out_of_range_t & oor_flag)
    {
        axis_type const & axis = *static_cast<axis_type const *>(axisptr.get());

        axis_value_type_traits avtt;
        axis_value_type_traits::value_ref_type value = axis_value_type_traits::dereference(avtt, value_ptr);

        if(is_extendable())
        {
            // The axis is extenable, so there are no under- and overflow bins.
            //std::cout << "Got value: " << value << std::endl;
            if(value < axis.min_)
            {
                //std::cout << "underflow: " << value << ", min = "<< data.min_ << std::endl;
                oor_flag = axis::OOR_UNDERFLOW;
                return -1;
            }
            intptr_t const idx = (value - axis.min_)/axis.bin_width_;
            if(idx >= axis.n_bins_)
            {
                //std::cout << "overflow: " << value << ", idx = "<< idx << std::endl;
                oor_flag = axis::OOR_OVERFLOW;
                return -1;
            }
            //std::cout << "value " << value << " at " << idx << std::endl;
            oor_flag = axis::OOR_NONE;
            return idx;
        }
        else
        {
            // The axis is not extenable, so there are under- and overflow bins
            // available.
            if(value < axis.underflow_edge_)
            {
                // The value falls even left to the underflow bin.
                oor_flag = axis::OOR_UNDERFLOW;
                return -1;
            }
            if(value < axis.min_)
            {
                // The value falls into the underflow bin.
                oor_flag = axis::OOR_NONE;
                return 0;
            }
            intptr_t const idx = (value - axis.min_)/axis.bin_width_;
            if(idx < axis.n_bins_-2)
            {
                // The value falls into the axis range (excluding the overflow
                // bin).
                oor_flag = axis::OOR_NONE;
                return idx;
            }
            if(value < axis.overflow_edge_)
            {
                // The value falls into the overflow bin.
                oor_flag = axis::OOR_NONE;
                return axis.n_bins_-1;
            }

            // The value falls even right to the overflow bin.
            oor_flag = axis::OOR_OVERFLOW;
            return -1;
        }
    }

    // Determines the number of extra bins needed to the left (negative number
    // returned) or to the right (positive number returned) of the current axis
    // range.
    static
    intptr_t
    request_extension(boost::shared_ptr<Axis> & axisptr, char * value_ptr, axis::out_of_range_t const oor_flag)
    {
        axis_type & axis = *static_cast< axis_data_type *>(axisdata.get());

        axis_value_type_traits avtt;
        axis_value_type_traits::value_ref_type value = axis_value_type_traits::dereference(avtt, value_ptr);

        if(oor_flag == axis::OOR_UNDERFLOW)
        {
            intptr_t const n_extra_bins = std::ceil((std::abs(value - axis.min_) / axis.bin_width_));
            //std::cout << "request_autoscale (underflow): " << n_extra_bins << " extra bins." << std::endl<< std::flush;
            return -n_extra_bins;
        }
        else if(oor_flag == axis::OOR_OVERFLOW)
        {
            intptr_t const n_extra_bins = intptr_t((value - axis.min_)/axis.bin_width_) - (axis.n_bins_-1);
            //std::cout << "request_autoscale (overflow): " << n_extra_bins << " extra bins." << std::endl<< std::flush;
            return n_extra_bins;
        }

        return 0;
    }

    static
    void
    extend(boost::shared_ptr<Axis> & axisptr, intptr_t f_n_extra_bins, intptr_t b_n_extra_bins)
    {
        axis_type & axis = *static_cast< axis_type *>(axisptr.get());

        if(f_n_extra_bins > 0)
        {
            axis.n_bins_ += f_n_extra_bins;
            axis.min_    -= f_n_extra_bins * axis.bin_width_;
        }
        if(b_n_extra_bins > 0)
        {
            axis.n_bins_ += b_n_extra_bins;
        }
    }
};

// Meta function to select the correct axis class.
// In cases where the axis value type is boost::python::object, the
// ConstantBinWidthAxis template cannot be used due to its performed arithmetic
// operations. In those cases, we need to fall back to the GenericAxis template,
// which requires only the comparsion operator to be implemented for the
// boost::python::object object.
template <typename AxisValueType>
struct ConstantBinWidthAxis_selector
{
    typedef ConstantBinWidthAxis<AxisValueType>
            type;
};

template <typename AxisValueType>
struct GenericAxis_selector
{
    typedef GenericAxis<AxisValueType>
            type;
};

template <typename AxisValueType>
struct select_ConstantBinWidthAxis_type
{
    typedef typename boost::mpl::eval_if<
                boost::is_same<AxisValueType, boost::python::object>
              , GenericAxis_selector<AxisValueType>
              , ConstantBinWidthAxis_selector<AxisValueType>
              >::type
            type;
};

// In order to expose the ConstantBinWidthAxis to Python, we need a wrapper
// around the ConstantBinWidthAxis template to get rid of the template
// parameter to avoid several Python classes, one for each axis value type.
namespace py {

class constant_bin_width_axis
  : public Axis
{
  protected:
    // This is the actual axis object which is type sensitive.
    boost::shared_ptr<Axis> axis_;

  public:
    constant_bin_width_axis(
        bn::ndarray const & edges
      , std::string const & label
      , bool is_extendable=false
      , intptr_t extension_max_fcap=0
      , intptr_t extension_max_bcap=0
    )
    {
        // Create the ConstantBinWidthAxis<AxisValueType> object on the heap
        // and save a pointer to it. Then copy the API function pointers into
        // this Axis object. So we don't have to do type lookups whenever
        // calling an API function.
        bool axis_dtype_is_supported = false;
        bn::dtype axis_dtype = edges.get_dtype();
        #define NDHIST_AXES_PY_AXIS_DTYPE_SUPPORT(r, data, AXIS_VALUE_TYPE)           \
            if(bn::dtype::equivalent(axis_dtype, bn::dtype::get_builtin<AXIS_VALUE_TYPE>()))\
            {                                                                         \
                if(axis_dtype_is_supported)                                           \
                {                                                                     \
                    std::stringstream ss;                                             \
                    ss << "The axis value data type is supported by more than one "   \
                       << "possible C++ data type! This is an internal error!";       \
                    throw TypeError(ss.str());                                        \
                }                                                                     \
                axis_ = boost::shared_ptr< select_ConstantBinWidthAxis_type<AXIS_VALUE_TYPE>::type >(new select_ConstantBinWidthAxis_type<AXIS_VALUE_TYPE>::type(edges, label, is_extendable, extension_max_fcap, axis_extension_max_bcap));\
                axis_dtype_is_supported = true;                                       \
            }
        BOOST_PP_SEQ_FOR_EACH(NDHIST_AXES_PY_AXIS_DTYPE_SUPPORT, ~, NDHIST_SUPPORTED_AXIS_VALUE_TYPES)
        #undef NDHIST_AXES_PY_AXIS_DTYPE_SUPPORT

        // Wrap the axis_ Axis object.
        wrap_axis(*axis_);
    }
};

}//namespace py

}//namespace axes
}//namespace ndhist

#endif // NDHIST_AXES_CONSTANT_BIN_WIDTH_AXIS_HPP_INCLUDED
