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
#include <boost/python/import.hpp>

namespace bp = boost::python;

namespace ndhist {

namespace stats {

void register_mean();
void register_moment();

}// namespace stats

void register_stats_module()
{
    bp::object stats_module = bp::import("ndhist.stats");
    {
        bp::scope stats_module_scope(stats_module);

        stats::register_mean();
        stats::register_moment();
    }
}

}// namespace ndhist
