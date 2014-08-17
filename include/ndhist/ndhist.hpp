#ifndef NDHIST_NDHIST_H_INCLUDED
#define NDHIST_NDHIST_H_INCLUDED 1

#include <iostream>
#include <stdint.h>

#include <vector>
#include <iostream>

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/detail/error.hpp>
#include <ndhist/detail/nddatarray.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

class ndhist
{
  public:
    ndhist(
        bn::ndarray const & shape
      , bp::list const & edges
      , bn::dtype const & dt
    )
    {
        if(shape.get_nd() != 1)
        {
            throw detail::error(
                "The shape array must be 1-dimensional!");
        }
        if(shape.get_size() != bp::len(edges))
        {
            throw detail::error(
                "The size of the shape array and the length of the edges list "
                "must be equal!");
        }

        const intptr_t nd = shape.get_size();
        // Construct the bin content array.

    }

  private:
    ndhist() {};

    /// The bin contents.
    boost::shared_ptr<detail::nddatarray> bc_ptr_;

    /// The vector of the edges arrays.
    boost::shared_ptr< std::vector<detail::nddatarray> > edges_ptr_;
};

}// namespace ndhist

#endif
