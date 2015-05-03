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
void register_median();
void register_moment();
void register_skewness();
void register_std();
void register_var();

}// namespace stats

void register_stats_module()
{
    bp::object stats_module = bp::import("ndhist.stats");
    {
        bp::scope stats_module_scope(stats_module);

        stats::register_mean();
        stats::register_median();
        stats::register_moment();
        stats::register_skewness();
        stats::register_std();
        stats::register_var();
    }
}

}// namespace ndhist
