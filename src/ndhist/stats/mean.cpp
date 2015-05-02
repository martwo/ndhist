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
#include <boost/python.hpp>

#include <ndhist/stats/moment.hpp>
#include <ndhist/stats/mean.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace stats {
namespace py {

bp::object
mean(ndhist const & h, bp::object const & axis)
{
    return moment(h, 1, axis);
}

}// namespace py
}// namespace stats
}// namespace ndhist
