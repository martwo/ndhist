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
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/import.hpp>

#include <ndhist/storage.hpp>

namespace bp = boost::python;

namespace ndhist {

void
histsave(
    ndhist const & h
  , std::string const & f
  , std::string const & where
  , std::string const & name
  , bool const overwrite
)
{
    bp::object storage = bp::import("ndhist.storage");
    bp::object py_histsave = storage.attr("histsave");
    py_histsave(boost::cref(h), f, where, name, overwrite);
}

boost::shared_ptr<ndhist>
histload(
    std::string const & f
  , std::string const & histgroup
)
{
    bp::object storage = bp::import("ndhist.storage");
    bp::object py_histload = storage.attr("histload");
    return bp::extract< boost::shared_ptr<ndhist> >(py_histload(f, histgroup));
}

}// namespace ndhist
