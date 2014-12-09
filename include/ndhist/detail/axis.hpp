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
#ifndef NDHIST_DETAIL_AXIS_HPP_INCLUDED
#define NDHIST_DETAIL_AXIS_HPP_INCLUDED 1

#include <boost/function.hpp>
#include <boost/python.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/detail/ndarray_storage.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

struct ndhist;

namespace detail {

struct AxisData;

struct Axis
{
    boost::function<intptr_t (boost::shared_ptr<AxisData>, char *)> get_bin_index_fct;
    boost::shared_ptr<AxisData> data_;
};

struct AxisData
{};

}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_AXIS_HPP_INCLUDED
