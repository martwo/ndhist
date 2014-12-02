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




namespace bp = boost::python;
namespace bn = boost::numpy;
namespace ds = bn::dstream;

namespace ndhist {

void register_error_types();
void register_ndhist();
void register_ndtable();



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

    return r;
}

static
std::vector< std::vector<int> >
power_series( std::vector<int> v )
{
    std::cout << "power_series called." << std::endl;
    std::vector< std::vector<int> > r(3);
    for(int exp=0; exp<3; ++exp)
    {
        r[exp] = std::vector<int>(v.size());
        for(int i=0; i<v.size(); ++i)
        {
            r[exp][i] = std::pow(v[i], exp);
        }
    }
    std::cout << "power_series returns." << std::endl;
    return r;
}

static
void multidimvectorarg(std::vector< std::vector< std::vector<int> > > v)
{
    for(int i=0; i<v.size(); ++i)
    {
        for(int j=0; j<v[i].size(); ++j)
        {
            for(int k=0; k<v[i][j].size(); ++k)
            {
                std::cout << "v["<<i<<"]["<<j<<"]["<<k<<"] = "<<v[i][j][k] << std::endl;
            }
        }
    }
}

static
void nd1vectorarg(std::vector< int > v)
{
    for(int i=0; i<v.size(); ++i)
    {
        std::cout << "v["<<i<<"] = "<<v[i] << std::endl;
    }
}

static
bp::object obj_return(int v)
{
    std::stringstream ss;
    ss << "v = " << v << std::endl;
    return bp::str(ss.str());
}

static
int scalar_return(int v)
{
    return v*v;
}

}// namespace ndhist

BOOST_PYTHON_MODULE(ndhist)
{
    boost::numpy::initialize();
    ndhist::register_error_types();
    ndhist::register_ndhist();
    ndhist::register_ndtable();

    ds::def("testfct", &ndhist::testfct
        , (bp::arg("t"))
        , "Doc"
        //, ( ds::array<2>() >> ds::array<2>() )
        , ds::wiring::generalized_wiring_model_selector()
    );

    ds::def("multidimvectorarg", &ndhist::multidimvectorarg
        , (bp::arg("v"))
        , "Doc"
        , ds::wiring::generalized_wiring_model_selector()
    );

    ds::def("nd1vectorarg", &ndhist::nd1vectorarg
        , (bp::arg("v"))
        , "Doc"
        , ds::wiring::generalized_wiring_model_selector()
    );

    ds::def("power_series", ndhist::power_series
        , (bp::arg("v"))
        , "Doc"
        , (ds::array<ds::dim::I>() >> ( ds::array<ds::dim::I>(), ds::array<ds::dim::I>(), ds::array<ds::dim::I>() ))
        , ds::wiring::generalized_wiring_model_selector()
    );

    ds::def("obj_return", ndhist::obj_return, (bp::arg("v")), "Doc");

    ds::def("scalar_return", ndhist::scalar_return, (bp::arg("v")), "Doc");


}
