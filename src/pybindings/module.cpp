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

#include <boost/numpy.hpp>

namespace ndhist {

void register_error_types();
void register_axis();
void register_ndhist();
void register_ndtable();

}// namespace ndhist

BOOST_PYTHON_MODULE(ndhist)
{
    boost::numpy::initialize();
    ndhist::register_error_types();
    ndhist::register_axis();
    ndhist::register_ndhist();
    ndhist::register_ndtable();
}
