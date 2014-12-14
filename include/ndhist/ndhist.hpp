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
#ifndef NDHIST_NDHIST_HPP_INCLUDED
#define NDHIST_NDHIST_HPP_INCLUDED 1

#include <iostream>
#include <stdint.h>

#include <vector>
#include <iostream>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>

#include <ndhist/error.hpp>
#include <ndhist/detail/axis.hpp>
#include <ndhist/detail/ndarray_storage.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

class ndhist
{
  public:

    /** Constructor for specifying a specific shape and bins (with (non-equal)
     *  sized widths).
     *
     *  The shape ndarray holds must be 1-dim. with ND
     *  integer elements, specifying the number of bins for each dimension.
     *
     *  The edges list must contain ND 1-dim. ndarray objects with shape[i]+1
     *  elements in ascending order, beeing the bin edges. The different
     *  dimensions can have different edge types, e.g. integer or float, or any
     *  other Python type, even entire objects.
     *
     *  The dt dtype object defines the data type for the bin contents. For a
     *  histogram this is usually an integer or float type.
     */
    ndhist(
        bn::ndarray const & shape
      , bp::list const & edges
      , bn::dtype const & dt
      , bp::object const & bc_class = bp::object()
    );

    virtual ~ndhist() {}

    bn::ndarray
    GetBinContentArray();

    /**
     * \brief Returns the ndarray holding the bin edges of the given axis.
     *        Note, that this is always a copy, since the edges are supposed
     *        to be readonly, because a re-edging of an already filled histogram
     *        does not make sense.
     */
    bn::ndarray
    get_edges_ndarray(intptr_t axis=0) const;

    /** Fills a given n-dimension value into the histogram's bin content array.
     *  On the Python side, the *ndvalue* is a numpy object array that might
     *  hold values of different types. The order of these types must match the
     *  types of the bin edges vector.
     */
    void Fill(bp::object const & ndvalue_obj, bp::object const & weight_obj);

    void handle_struct_array(bp::object const & arr_obj);

    inline
    std::vector< boost::shared_ptr<detail::Axis> > &
    GetAxes()
    {
        return axes_;
    }

    inline
    bn::ndarray &
    GetBCArray()
    {
        return *static_cast<bn::ndarray*>(&bc_arr_);
    }

    inline
    bn::ndarray const &
    GetBCArray() const
    {
        return *static_cast<bn::ndarray const *>(&bc_arr_);
    }

    inline
    int get_nd() const
    {
        return GetBCArray().get_nd();
    }

  private:
    ndhist() {};

    /** Constructs a ndarray_storage object for the bin contents. The extra
     *  front and back capacity will be set to zero.
     */
    static
    boost::shared_ptr<detail::ndarray_storage>
    ConstructBinContentStorage(bn::ndarray const & shape, bn::dtype const & dt);

    /** The bin contents.
     */
    boost::shared_ptr<detail::ndarray_storage> bc_;
    boost::python::object bc_arr_;

    std::vector< boost::shared_ptr<detail::Axis> > axes_;

    boost::function<void (ndhist &, bp::object const &, bp::object const &)> fill_fct_;
};

}// namespace ndhist

#endif // !NDHIST_NDHIST_HPP_INCLUDED
