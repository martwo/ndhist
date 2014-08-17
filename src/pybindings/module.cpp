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
#include <boost/numpy.hpp>
#include <boost/python.hpp>

namespace ndhist {

void register_ndhist();

}// namespace ndhist

BOOST_PYTHON_MODULE(ndhist)
{
    boost::numpy::initialize();
    ndhist::register_ndhist();
}
