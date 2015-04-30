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
#ifndef NDHIST_STORAGE_HPP_INCLUDED
#define NDHIST_STORAGE_HPP_INCLUDED 1

#include <string>

#include <boost/shared_ptr.hpp>

#include <ndhist/ndhist.hpp>

namespace ndhist {

/**
 * Saves the given ndhist object to the given file.
 *
 * @param h The ndhist object that should get stored.
 *
 * @param f The name of the file, with a propper file extension, so the file
 *     format can be determined.
 *     The following file extensions are supported for HDF files:
 *
 *       - ``".hdf"``
 *       - ``".h5"``
 *
 * @param where The parent group/location within the file where the histogram
 *     should get stored to.
 *
 * @param name The name of the data group for this histogram.
 *
 * @param overwrite Flag if an existing histogram of the same name should get
 *     overwritten (``true``) or not (``false``).
 *
 * @internal This function calls the Python function ``histsave`` implemented in
 *     the storage.py python module of the ndhist python package.
 */
void
histsave(
    ndhist const & h
  , std::string const & f
  , std::string const & where
  , std::string const & name
  , bool const overwrite=false
);

/**
 * Loads a ndhist object, that is stored within the given group within the
 * given file. The loaded ndhist object will live on the heap.
 *
 * @param f The name of the file, in which the histogram is stored. It must
 *     contain a propper file extension, so the file format can be determined.
 *     The following file extensions are supported for HDF files:
 *
 *       - ``".hdf"``
 *       - ``".h5"``
 *
 * @param histgroup The data group name within the file where the histogram is
 *     stored in. It's the group created by the histsave function.
 *
 * @internal This function calls the Python function ``histload`` implemented in
 *     the storage.py python module of the ndhist python package.
 */
boost::shared_ptr<ndhist>
histload(
    std::string const & f
  , std::string const & histgroup
);

}// namespace ndhist

#endif // !NDHIST_STORAGE_HPP_INCLUDED
