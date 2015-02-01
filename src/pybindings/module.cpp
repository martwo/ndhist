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

namespace axes {

void register_constant_bin_width_axis();

}//namespace axes

}//namespace ndhist

BOOST_PYTHON_MODULE(ndhist_core)
{
    boost::numpy::initialize();

    ndhist::register_error_types();
    ndhist::register_axis();
    ndhist::axes::register_constant_bin_width_axis();
    ndhist::register_ndhist();
    ndhist::register_ndtable();
}
