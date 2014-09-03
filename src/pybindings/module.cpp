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
#include <iostream>

#include <boost/python.hpp>

#include <boost/numpy.hpp>
#include <boost/numpy/dstream.hpp>
#include <boost/numpy/dstream/wiring/generalized_wiring_model.hpp>
//#include <ndhist/detail/fill_wiring_model.hpp>


namespace bp = boost::python;
namespace bn = boost::numpy;
namespace ds = bn::dstream;

namespace ndhist {

void register_error_types();
void register_ndhist();

static
std::vector<int> testfct(std::vector<bp::object> v)
{
    std::cout << "v.size = " << v.size() << std::endl;
    // Now get attr _v from each object.
    for (int i=0; i<v.size(); ++i)
    {

        bp::object r = bp::getattr(v[i], bp::str("_v"));
        double d = bp::extract<double>(r);
        std::cout << "d"<<i<<" = "<<d << ", ";
    }

    std::vector<int> r;
    for(int i=0; i<v.size(); ++i)
    {
        r.push_back(i);
    }
    r.push_back(300);
    return r;
}


}// namespace ndhist

BOOST_PYTHON_MODULE(ndhist)
{
    boost::numpy::initialize();
    ndhist::register_error_types();
    ndhist::register_ndhist();

    ds::def("testfct", &ndhist::testfct
        , (bp::arg("t"))
        , "Doc"
        //, ( ds::array<2>() >> ds::array<2>() )
        , ds::wiring::generalized_wiring_model_selector()
    );
}
