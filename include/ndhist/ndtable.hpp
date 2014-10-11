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
#ifndef NDHIST_NDTABLE_HPP_INCLUDED
#define NDHIST_NDTABLE_HPP_INCLUDED 1

#include <boost/shared_ptr.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/dtype.hpp>

#include <ndhist/detail/ndarray_storage.hpp>

namespace ndhist {

/** The ndtable class provides a container for holding the tabulated values of
 *  a multi-dimensional mathematical function. It is basically just a ndarray.
 *  It manages a ndarray_storage object.
 */
class ndtable
{
  public:
    ndtable(
        boost::numpy::ndarray const & shape
      , boost::numpy::dtype const & dt
    );

    virtual ~ndtable() {}

  private:
    /** Constructs a ndarray_storage object for the ndtable's data. The extra
     *  front and back capacity will be set to zero.
     */
    static
    boost::shared_ptr<detail::ndarray_storage>
    ConstructDataStorage(boost::numpy::ndarray const & shape, boost::numpy::dtype const & dt);

    boost::shared_ptr<detail::ndarray_storage> data_;
};

}// namespace ndhist

#endif // ! NDHIST_NDTABLE_HPP_INCLUDED
