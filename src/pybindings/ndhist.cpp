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
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

#include <ndhist/ndhist.hpp>

namespace bp = boost::python;

namespace ndhist {

void register_ndhist()
{
    bp::class_<ndhist, boost::shared_ptr<ndhist> >("ndhist"
        , "The ndhist class provides a multi-dimensional histogram class."
        , bp::init<
            bn::ndarray const &
          , bp::list const &
          , bn::dtype const &
          >(
          ( bp::arg("nbins")
          , bp::arg("edges")
          , bp::arg("dtype")
          )
          )
        )

        .add_property("bc", bp::make_function(&ndhist::GetBinContentArray, bp::with_custodian_and_ward_postcall<0,1>())
            , "The ndarray holding the bin contents.")
            // We keep the ndhist object (i.e. "1") alive as long as the
            // returned ndarray (i.e. "0") is alive.
    ;
}

}// namespace ndhist
