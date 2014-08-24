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
#include <boost/python.hpp>
#include <boost/python/ptr.hpp>

#include <ndhist/ndhist.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

bn::ndarray
ndhist::
GetBinContentArray()
{
    bp::object self(bp::ptr(this));
    return bc_.ConstructNDArray(self);
}

}//namespace ndhist
