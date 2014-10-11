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
#include <boost/python/list.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>


#include <ndhist/error.hpp>
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
        boost::numpy::ndarray const & shape
      , boost::python::list const & edges
      , boost::numpy::dtype const & dt
    );

    virtual ~ndhist() {}

    boost::numpy::ndarray
    GetBinContentArray();

    boost::numpy::ndarray
    GetEdgesArray(int axis=0);

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

    /** The vector of the edges arrays.
     */
    std::vector< boost::shared_ptr<detail::ndarray_storage> > edges_;
};

}// namespace ndhist

#endif // !NDHIST_NDHIST_HPP_INCLUDED
