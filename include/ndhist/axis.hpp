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

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>
#include <boost/python/slice.hpp>

#include <boost/numpy.hpp>

#include <ndhist/detail/axis.hpp>
#include <ndhist/error.hpp>
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
      , has_oor_bins_(false)
      , is_extendable_(false)
      , extension_max_fcap_(0)
      , extension_max_bcap_(0)
    {}

    Axis(
        size_t nbins
      , boost::numpy::dtype const & dt
      , std::string const & label
      , std::string const & name
      , bool has_oor_bins=true
      , bool is_extendable=false
      , intptr_t extension_max_fcap=0
      , intptr_t extension_max_bcap=0
    )
      : dt_(dt)
      , label_(label)
      , name_(name)
      , has_oor_bins_(has_oor_bins)
      , is_extendable_(is_extendable)
      , extension_max_fcap_(extension_max_fcap)
      , extension_max_bcap_(extension_max_bcap)
    {
        size_t const nedges = nbins + 1;
        if(is_extendable_)
        {
            if(nedges < 2)
            {
                std::stringstream ss;
                ss << "The axis \""<< name_ <<"\" is supposed to be "
                   << "extendable, so the minimum number of edges provided "
                   << "must be 2! Currently it is "<<nedges<<"!";
                throw ValueError(ss.str());
            }
        }
        else
        {
            if(has_oor_bins_)
            {
                if(nedges < 4)
                {
                    std::stringstream ss;
                    ss << "The axis \""<< name_ <<"\" is supposed to be "
                       << "non-extendable, so the minimum number of edges provided "
                       << "must be 4, which includes the under- and overflow bins! "
                       << "Currently it is "<<nedges<<"!";
                    throw ValueError(ss.str());
                }
            }
            else
            {
                if(nedges < 2)
                {
                    std::stringstream ss;
                    ss << "The axis \""<< name_ <<"\" is supposed to be "
                       << "non-extendable with no out-of-range bins. "
                       << "So the minimum number of edges provided must be 2. "
                       << "Currently it is "<<nedges<<"!";
                    throw ValueError(ss.str());
                }
            }
        }
    }

    virtual ~Axis()
    {
        std::cout << "Destruct Axis"<<std::endl<<std::flush;
    }

    /** Returns a reference to the dtype object of the axis values.
     */
    boost::numpy::dtype const &
    get_dtype() const
    {
        return get_axis_base().dt_;
    }

    std::string
    py_get_name() const
    {
        return get_axis_base().name_;
    }

    void
    py_set_name(std::string const & name)
    {
        get_axis_base().name_ = name;
    }

    bool
    py_has_oor_bins() const
    {
        return has_oor_bins();
    }

    /**
     * @brief Returns ``true`` if the axis is extendable.
     */
    bool
    py_is_extendable() const
    {
        return is_extendable();
    }

    intptr_t
    py_get_n_bins() const
    {
        return get_axis_base().get_n_bins_fct_(get_axis_base());
    }

    /** Wraps the given other Axis object. This means, it sets the wrapped_axis_
     *  member to the wrapped Axis object, which is returned whenever the
     *  get_axis_base method is called.
     */
    void
    wrap_axis(boost::shared_ptr<Axis> & wrapped_axis)
    {
        wrapped_axis_ = wrapped_axis;
    }

    inline
    Axis &
    get_axis_base()
    {
        return (wrapped_axis_ == NULL ? *this : *wrapped_axis_);
    }

    inline
    Axis const &
    get_axis_base() const
    {
        return (wrapped_axis_ == NULL ? *this : *wrapped_axis_);
    }

    inline
    bool
    has_oor_bins() const
    {
        return get_axis_base().has_oor_bins_;
    }

    inline
    bool
    is_extendable() const
    {
        return get_axis_base().is_extendable_;
    }

    inline
    intptr_t
    get_bin_index(char * value_ptr, axis::out_of_range_t & oor_flag) const
    {
        return get_axis_base().get_bin_index_fct_(get_axis_base(), value_ptr, oor_flag);
    }

    inline
    intptr_t
    get_extension_max_fcap() const
    {
        return get_axis_base().extension_max_fcap_;
    }

    inline
    intptr_t *
    get_extension_max_fcap_ptr()
    {
        return & get_axis_base().extension_max_fcap_;
    }

    inline
    intptr_t
    get_extension_max_bcap() const
    {
        return get_axis_base().extension_max_bcap_;
    }

    inline
    intptr_t *
    get_extension_max_bcap_ptr()
    {
        return & get_axis_base().extension_max_bcap_;
    }

    inline
    std::string const &
    get_label() const
    {
        return get_axis_base().label_;
    }

    inline
    std::string &
    get_label()
    {
        return get_axis_base().label_;
    }

    inline
    intptr_t
    get_n_bins() const
    {
        return get_axis_base().get_n_bins_fct_(get_axis_base());
    }

    inline
    std::string const &
    get_name() const
    {
        return get_axis_base().name_;
    }

    inline
    std::string &
    get_name()
    {
        return get_axis_base().name_;
    }

    inline
    boost::numpy::ndarray
    get_binedges_ndarray() const
    {
        return get_axis_base().get_binedges_ndarray_fct_(get_axis_base());
    }

    inline
    boost::numpy::ndarray
    get_bincenters_ndarray() const
    {
        return get_axis_base().get_bincenters_ndarray_fct_(get_axis_base());
    }

    inline
    intptr_t
    request_extension(char * const value_ptr, axis::out_of_range_t const oor_flag)
    {
        return get_axis_base().request_extension_fct_(get_axis_base(), value_ptr, oor_flag);
    }

    inline
    void
    extend(intptr_t const f_n_extra_bins, intptr_t const b_n_extra_bins)
    {
        get_axis_base().extend_fct_(get_axis_base(), f_n_extra_bins, b_n_extra_bins);
    }

    inline
    boost::shared_ptr<Axis>
    create_axis_slice(intptr_t const start, intptr_t const stop, intptr_t const step, intptr_t const nbins)
    {
        return get_axis_base().create_axis_slice_fct_(get_axis_base(), start, stop, step, nbins);
    }

    template <class AxisType>
    static
    boost::numpy::ndarray
    get_bincenters_ndarray(Axis const & axisbase)
    {
        boost::numpy::ndarray const binedges = axisbase.get_binedges_ndarray();
        intptr_t shape[1];
        shape[0] = binedges.shape(0) - 1;
        boost::numpy::ndarray bincenters = boost::numpy::empty(1, shape, boost::numpy::dtype::get_builtin<typename AxisType::axis_value_type>());
        typedef boost::numpy::iterators::flat_iterator< boost::numpy::iterators::single_value<typename AxisType::axis_value_type> >
                iter_t;
        iter_t binedges_it0(binedges);
        iter_t binedges_it1(binedges);
        ++binedges_it1;
        iter_t bincenters_it(bincenters);
        while(! bincenters_it.is_end())
        {
            typename AxisType::axis_value_type v = *binedges_it0 + *binedges_it1;
            v *= 0.5;
            bincenters_it.set_value(v);
            ++bincenters_it;
            ++binedges_it0;
            ++binedges_it1;
        }
        return bincenters;
    }

    template <class AxisType>
    static
    boost::shared_ptr<Axis>
    create_axis_slice(Axis const & axisbase, intptr_t const start, intptr_t const stop, intptr_t const & step, intptr_t const nbins)
    {

        std::cout << "create_axis_slice"<<std::endl;
        std::cout << "start = "<<start<<", stop = "<<stop << ", step = "<<step<<", nbins="<<nbins << std::endl;
        // Construct the edges array.
        typedef boost::numpy::iterators::flat_iterator< boost::numpy::iterators::single_value<typename AxisType::axis_value_type> >
                iter_t;

        boost::numpy::ndarray alledges = axisbase.get_binedges_ndarray();
        iter_t begin(alledges);
        iter_t end(begin);
        std::advance(end, alledges.get_size()-1);
        boost::python::slice sl(start, stop, step);
        boost::python::slice::range<iter_t> r = sl.get_indices<iter_t>(begin, end);
        // Define the shape of the edges array, it is the number of bins
        // plus the upper edge at the end.
        std::vector<intptr_t> const shape(1, nbins + 1);
        boost::numpy::ndarray edges = boost::numpy::empty(shape, alledges.get_dtype());
        iter_t it(edges);
        while(r.start != r.stop)
        {
            // If the step is negative, we need to use the upper edge, instead
            // of the lower edge.
            it.set_value( *(r.start + (step < 0)) );
            ++it;
            std::advance(r.start, r.step);
        }
        it.set_value( *(r.start + (step < 0)) );
        // Add the upper (step == +1) or lower edge (step == -1).
        ++it;
        std::advance(r.start, r.step);
        it.set_value( *r.start );

        // Construct the axis object.
        // Since it defines a data view, no out-of-range bins are supported,
        // it cannot be extended, and it can't have extra capacity.
        boost::shared_ptr<Axis> axis(new AxisType(
            edges
          , axisbase.get_label()
          , axisbase.get_name()
          , /*has_oor_bins=*/false
          , /*is_extendable=*/false
          , /*extension_max_fcap=*/0
          , /*extension_max_bcap=*/0
        ));

        return axis;
    }

  protected:
    /** In case this Axis object wraps an other Axis object, this stores the
     *  pointer to it.
     */
    boost::shared_ptr<Axis> wrapped_axis_;

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

    /** Flag if the axis has out-or-range bins. This is usually true for a
     *  histogram, but is false, if the histogram is a slice (view) of an
     *  original histogram.
     */
    bool has_oor_bins_;

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
    boost::function<intptr_t (Axis const &, char * const, axis::out_of_range_t &)>
        get_bin_index_fct_;

    /** This function is supposed to return (a copy of) the edges array
     *  (including the possible under- and overflow bins) as a
     *  boost::numpy::ndarray object.
     */
    boost::function<boost::numpy::ndarray (Axis const &)>
        get_binedges_ndarray_fct_;

    /** This function is supposed to return (a copy of) the bincenters array
     *  (including the possible under- and overflow bins) as a
     *  boost::numpy::ndarray object.
     */
    boost::function<boost::numpy::ndarray (Axis const &)>
        get_bincenters_ndarray_fct_;

    /** This function is supposed to return the number of bins of the axis
     *  (including the possible under- and overflow bins).
     */
    boost::function<intptr_t (Axis const &)>
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
    boost::function<intptr_t (Axis const &, char * const, axis::out_of_range_t const)>
        request_extension_fct_;

    /** This function is supposed to extend the axis by the given number of bins
     *  to the left and the right of the axis, respectively.
     *  Note: This function is only called, when the axis is extendable, thus
     *        there are no under- and overflow bins defined in those cases.
     */
    boost::function<void (Axis &, intptr_t const, intptr_t const)>
        extend_fct_;

    /**
     * @brief This function is supposed to create an axis of the same type as
     *     this axis but is slice of this axis. The slice is defined by start,
     *     stop, and step, nbins.
     *     The value interval of start is [0,n).
     *     The value interval of stop is [-1,n], where n is the number of
     *     (current) bins of this axis.
     *     Step can only be +1 or -1.
     *     Nbins is the number of bins of the resulting axis. Thus, the number
     *     of edges of the resulting axis must be nbins+1.
     */
    boost::function<boost::shared_ptr<Axis> (Axis const &, intptr_t const, intptr_t const, intptr_t const, intptr_t const)>
        create_axis_slice_fct_;
};

/** The axis_pyinterface template provides a boost::python::def_visitor for
 *  exposing the axis interface automatically to Python.
 */
template <class AxisType>
class axis_pyinterface
  : public boost::python::def_visitor< axis_pyinterface<AxisType> >
{
  public:
    axis_pyinterface()
    {}

  private:
    friend class boost::python::def_visitor_access;

    template <class ClassT>
    void visit(ClassT & cls) const
    {
        cls.add_property("name"
            , &AxisType::py_get_name
            , &AxisType::py_set_name
            , "The name of the axis. It is the name of the column in the "
              "structured ndarray, when filling values via a structured "
              "ndarray."
        );
        cls.add_property("has_oor_bins"
            , &AxisType::py_has_oor_bins
            , "Flag if the axis has out-of-range bins. This is usually true "
              "for a non-extendable axis, but is false for axis of slice "
              "histograms."
        );
        cls.add_property("is_extendable"
            , &AxisType::py_is_extendable
            , "Flag if the axis is extendable (True) or not (False)."
        );
        cls.add_property("nbins"
            , &AxisType::py_get_n_bins
            , "The number of bins this axis has."
        );
    }
};


/** The PyExtendableAxisWrapper template provides a wrapper for an axis template
 *  that depends on the axis value type and that is, in principle, extendable.
 *  In order to expose that axis template to Python, we need to get rid of the
 *  axis value type template parameter. The PyExtendableAxisWrapper will do
 *  that.
 *
 *  The requirement on the AxisTypeTemplate is that it has a member type named
 *  ``type`` which is the type of the to-be-wrapped axis object class.
 *  Furthermore, a constructor with the following call signature must be
 *  defined:
 *
 *      boost::numpy::ndarray const & edges
 *    , std::string const & label
 *    , std::string const & name
 *    , bool is_extendable
 *    , intptr_t extension_max_fcap
 *    , intptr_t extension_max_bcap
 */
template <template <typename AxisValueType> class AxisTypeTemplate>
class PyExtendableAxisWrapper
  : public Axis
{
  public:
    PyExtendableAxisWrapper(
        boost::numpy::ndarray const & edges
      , std::string const & label=std::string("")
      , std::string const & name=std::string("")
      , bool has_oor_bins=true
      , bool is_extendable=false
      , intptr_t extension_max_fcap=0
      , intptr_t extension_max_bcap=0
    )
    {
        // Create the AxisTypeTemplate<AxisValueType> object on the heap
        // and save a pointer to it. Then wrap this Axis object around the
        // created axis object. So we don't have to do type lookups whenever
        // calling an API function.
        boost::shared_ptr< Axis > wrapped_axis_ptr;
        bool axis_dtype_is_supported = false;
        boost::numpy::dtype axis_dtype = edges.get_dtype();
        #define NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT(r, data, AXIS_VALUE_TYPE)   \
            if(boost::numpy::dtype::equivalent(axis_dtype, boost::numpy::dtype::get_builtin<AXIS_VALUE_TYPE>()))\
            {                                                                   \
                if(axis_dtype_is_supported)                                     \
                {                                                               \
                    std::stringstream ss;                                       \
                    ss << "The axis value data type is supported by more than " \
                       << "one possible C++ data type! This is an internal "    \
                       << "error!";                                             \
                    throw TypeError(ss.str());                                  \
                }                                                               \
                wrapped_axis_ptr = boost::shared_ptr< typename AxisTypeTemplate<AXIS_VALUE_TYPE>::type >(new typename AxisTypeTemplate<AXIS_VALUE_TYPE>::type(edges, label, name, has_oor_bins, is_extendable, extension_max_fcap, extension_max_bcap));\
                axis_dtype_is_supported = true;                                 \
            }
        BOOST_PP_SEQ_FOR_EACH(NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_AXIS_VALUE_TYPES)
        #undef NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT

        // Wrap the axis_ Axis object.
        wrap_axis(wrapped_axis_ptr);
    }
};

/** The PyNonExtendableAxisWrapper template provides a wrapper for an axis
 *  template that depends on the axis value type and that is not extendable.
 *  In order to expose that axis template to Python, we need to get rid of the
 *  axis value type template parameter. The PyExtendableAxisWrapper will do
 *  that.
 *
 *  The requirement on the AxisTypeTemplate is that it has a member type named
 *  ``type`` which is the type of the to-be-wrapped axis object class.
 *  Furthermore, a constructor with the following call signature must be
 *  defined:
 *
 *      boost::numpy::ndarray const & edges
 *    , std::string const & label
 *    , std::string const & name
 */
template <template <typename AxisValueType> class AxisTypeTemplate>
class PyNonExtendableAxisWrapper
  : public Axis
{
  public:
    PyNonExtendableAxisWrapper(
        boost::numpy::ndarray const & edges
      , std::string const & label=std::string("")
      , std::string const & name=std::string("")
      , bool has_oor_bins=true
    )
    {
        // Create the AxisTypeTemplate<AxisValueType> object on the heap
        // and save a pointer to it. Then wrap this Axis object around the
        // created axis object. So we don't have to do type lookups whenever
        // calling an API function.
        boost::shared_ptr< Axis > wrapped_axis_ptr;
        bool axis_dtype_is_supported = false;
        boost::numpy::dtype axis_dtype = edges.get_dtype();
        #define NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT(r, data, AXIS_VALUE_TYPE)   \
            if(boost::numpy::dtype::equivalent(axis_dtype, boost::numpy::dtype::get_builtin<AXIS_VALUE_TYPE>()))\
            {                                                                   \
                if(axis_dtype_is_supported)                                     \
                {                                                               \
                    std::stringstream ss;                                       \
                    ss << "The axis value data type is supported by more than " \
                       << "one possible C++ data type! This is an internal error!";\
                    throw TypeError(ss.str());                                  \
                }                                                               \
                wrapped_axis_ptr = boost::shared_ptr< typename AxisTypeTemplate<AXIS_VALUE_TYPE>::type >(new typename AxisTypeTemplate<AXIS_VALUE_TYPE>::type(edges, label, name, has_oor_bins));\
                axis_dtype_is_supported = true;                                 \
            }
        BOOST_PP_SEQ_FOR_EACH(NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_AXIS_VALUE_TYPES)
        #undef NDHIST_DETAIL_PY_AXIS_DTYPE_SUPPORT

        // Wrap the axis_ Axis object.
        wrap_axis(wrapped_axis_ptr);
    }
};

}//namespace ndhist

#endif // NDHIST_AXIS_HPP_INCLUDED
