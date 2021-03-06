/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <ndhist@martin-wolf.org>
 *
 * @brief This file defines the GenericAxis template for a histogram axis with
 *        irregular bin widths. Due to the irregularity of the bin widths, such
 *        an axis cannot be extendable.
 *
 * This file is distributed under the BSD 2-Clause Open Source License
 * (See LICENSE file).
 *
 */
#ifndef NDHIST_AXES_GENERIC_AXIS_HPP_INCLUDED
#define NDHIST_AXES_GENERIC_AXIS_HPP_INCLUDED 1

#include <algorithm>
#include <sstream>

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/iterators/value_type_traits.hpp>
#include <boost/numpy/iterators/flat_iterator.hpp>

#include <ndhist/axis.hpp>
#include <ndhist/detail/ndarray_storage.hpp>
#include <ndhist/ndhist.hpp>
#include <ndhist/error.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace axes {

namespace detail {

template <typename AxisValueType>
struct GenericAxisValueCompare
{
    static
    bool
    apply(AxisValueType const & value, AxisValueType const & edge)
    {
        return (value < edge);
    }
};

template <>
struct GenericAxisValueCompare<bp::object>
{
    static
    bool
    apply(bp::object const & value, bp::object const & edge)
    {
        bp::object edge_type(bp::handle<>(bp::borrowed(bp::downcast<PyTypeObject>(PyObject_Type(edge.ptr())))));
        if(! PyObject_TypeCheck(value.ptr(), (PyTypeObject*)edge_type.ptr()))
        {
            std::stringstream ss;
            ss << "The type of the value for an axis must be the same as the "
               << "type of the edges objects of that axis! "
               << "Otherwise comparison operators might be ill-defined.";
            throw TypeError(ss.str());
        }
        return (value < edge);
    }
};

}//namespace detail

template <typename AxisValueType>
class GenericAxis
  : public Axis
{
  public:
    typedef AxisValueType
            axis_value_type;

    typedef bn::iterators::single_value<axis_value_type>
            axis_value_type_traits;

    typedef Axis
            base;

    typedef GenericAxis<AxisValueType>
            type;

  protected:
    ::ndhist::detail::ndarray_storage edges_arr_storage_;
    bp::object edges_arr_;
    bn::iterators::flat_iterator< axis_value_type_traits > edges_arr_iter_;
    bn::iterators::flat_iterator< axis_value_type_traits > edges_arr_iter_end_;
    axis_value_type_traits avtt_;

  public:
    GenericAxis(
        bn::ndarray const & edges
      , std::string const & label
      , std::string const & name
      , bool has_underflow_bin
      , bool has_overflow_bin
    )
      : Axis(
            edges.get_size()-1
          , edges.get_dtype()
          , label
          , name
          , has_underflow_bin
          , has_overflow_bin
          , false // is_extendable
          , 0     // extension_max_fcap
          , 0     // extension_max_bcap
        )
    {
        init(edges);
    }

    // This constructor is used, when a specialized axis needs to fall back to
    // the GenericAxis class, e.g. because it does not support
    // boost::python::object axis value types.
    GenericAxis(
        bn::ndarray const & edges
      , std::string const & label
      , std::string const & name
      , bool has_underflow_bin
      , bool has_overflow_bin
      , bool     /*is_extendable*/
      , intptr_t /*extension_max_fcap*/
      , intptr_t /*extension_max_bcap*/
    )
      : Axis(
            edges.get_size()-1
          , edges.get_dtype()
          , label
          , name
          , has_underflow_bin
          , has_overflow_bin
          , false // is_extendable
          , 0     // extension_max_fcap
          , 0     // extension_max_bcap
        )
    {
        init(edges);
    }

    /**
     * Copy constructor.
     */
    GenericAxis(GenericAxis const & other)
      : Axis(other)
      , edges_arr_storage_((*static_cast<type const *>(&other.get_axis_base())).edges_arr_storage_)
      , edges_arr_((*static_cast<bn::ndarray const *>(&(*static_cast<type const *>(&other.get_axis_base())).edges_arr_)).deepcopy())
      , edges_arr_iter_(bn::iterators::flat_iterator< bn::iterators::single_value<axis_value_type> >(*static_cast<bn::ndarray *>(&edges_arr_), bn::detail::iter_operand::flags::READONLY::value))
      , edges_arr_iter_end_(edges_arr_iter_.end())
    {}

    inline
    void
    init(bn::ndarray const & edges)
    {
        // Set up the axis's function pointers.
        create_fct_                     = &type::create;
        get_bin_index_fct_              = &type::get_bin_index;
        get_binedges_ndarray_fct_       = &type::get_binedges_ndarray;
        get_lower_binedges_ndarray_fct_ = &base::get_lower_binedges_ndarray<type>;
        get_upper_binedges_ndarray_fct_ = &base::get_upper_binedges_ndarray<type>;
        get_bincenters_ndarray_fct_     = &base::get_bincenters_ndarray<type>;
        get_binwidths_ndarray_fct_      = &base::get_binwidths_ndarray<type>;
        get_n_bins_fct_                 = &type::get_n_bins;
        request_extension_fct_          = NULL;
        extend_fct_                     = NULL;
        create_axis_slice_fct_          = &base::create_axis_slice<type>;
        deepcopy_fct_                   = &type::deepcopy;

        intptr_t const nbins = edges.get_size() - 1;
        if(nbins < 1)
        {
            std::stringstream ss;
            ss << "The edges array need to have at least 2 elements! But it "
               << "contains only "<<edges.get_size()<<" edge values!";
            throw ValueError(ss.str());
        }
        std::vector<intptr_t> shape(1, nbins+1);
        std::vector<intptr_t> front_capacity(1, 0);
        std::vector<intptr_t> back_capacity(1, 0);
        edges_arr_storage_ = ::ndhist::detail::ndarray_storage(edges.get_dtype(), shape, front_capacity, back_capacity);
        // Copy the data from the user provided edge array to the storage array.
        edges_arr_ = edges_arr_storage_.construct_ndarray(edges_arr_storage_.get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        bn::ndarray & arr = *static_cast<bn::ndarray*>(&edges_arr_);
        if(! bn::copy_into(arr, edges))
        {
            // TODO: Get the error string from the already set BP error.
            std::stringstream ss;
            ss << "Could not copy edge array into internal storage!";
            throw MemoryError(ss.str());
        }
        // Initialize a flat iterator over the axis edges.
        edges_arr_iter_ = bn::iterators::flat_iterator< bn::iterators::single_value<axis_value_type> >(arr, bn::detail::iter_operand::flags::READONLY::value);
        edges_arr_iter_end_ = edges_arr_iter_.end();
    }

    static
    boost::shared_ptr<Axis>
    create(
        boost::numpy::ndarray const & edges
      , std::string const & label
      , std::string const & name
      , bool has_underflow_bin
      , bool has_overflow_bin
      , bool is_extendable
      , intptr_t extension_max_fcap
      , intptr_t extension_max_bcap
    )
    {
        return boost::shared_ptr<Axis>(new GenericAxis(
            edges
          , label
          , name
          , has_underflow_bin
          , has_overflow_bin
          , is_extendable
          , extension_max_fcap
          , extension_max_bcap
        ));
    }

    static
    intptr_t
    get_bin_index(Axis const & axisbase, char * value_ptr, axis::out_of_range_t & oor_flag)
    {
        type & axis = *static_cast<type *>(const_cast<Axis *>( &axisbase ));

        typename axis_value_type_traits::value_ref_type value = axis_value_type_traits::dereference(axis.avtt_, value_ptr);

        // We know that edges is 1-dimensional by construction and the edges are
        // ordered ascedently. Also we know that the value type of the edges is
        // AxisValueType. So we can use the std::upper_bound binary search for
        // getting the upper edge for the given value.
        axis.edges_arr_iter_.reset();
        bn::iterators::flat_iterator< axis_value_type_traits > ub = std::upper_bound(axis.edges_arr_iter_, axis.edges_arr_iter_end_, value, &detail::GenericAxisValueCompare<axis_value_type>::apply);
        if(ub == axis.edges_arr_iter_end_)
        {
            // Overflow.
            oor_flag = axis::OOR_OVERFLOW;
            return -1;
        }
        intptr_t const idx = ub.get_iter_index();
        if(idx == 0)
        {
            // Underflow. ub points to the first element.
            oor_flag = axis::OOR_UNDERFLOW;
            return -1;
        }
        oor_flag = axis::OOR_NONE;
        return idx - 1;
    }

    static
    bn::ndarray
    get_binedges_ndarray(Axis const & axisbase)
    {
        type const & axis = *static_cast<type const *>(&axisbase);
        bn::ndarray const & edges_arr = *static_cast<bn::ndarray const *>(&axis.edges_arr_);
        return edges_arr.copy();
    }

    static
    intptr_t
    get_n_bins(Axis const & axisbase)
    {
        type const & axis = *static_cast<type const *>(&axisbase);
        bn::ndarray const & edges_arr = *static_cast<bn::ndarray const *>(&axis.edges_arr_);
        return edges_arr.get_size() - 1;
    }

    static
    boost::shared_ptr<Axis>
    deepcopy(Axis const & axisbase)
    {
        type const & axis = *static_cast< type const *>(&axisbase);
        return boost::shared_ptr<Axis>(new GenericAxis(axis));
    }
};

template <typename AxisValueType>
struct GenericAxis_selector
{
    typedef GenericAxis<AxisValueType>
            type;
};

namespace py {

typedef PyNonExtendableAxisWrapper<GenericAxis>
        generic_axis;

}//namespace py

}//namespace axes
}//namespace ndhist

#endif // !NDHIST_AXES_GENERIC_AXIS_HPP_INCLUDED
