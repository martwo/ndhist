#ifndef NDHIST_NDHIST_H_INCLUDED
#define NDHIST_NDHIST_H_INCLUDED 1

#include <iostream>
#include <stdint.h>

#include <vector>
#include <iostream>

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>


#include <ndhist/error.hpp>
#include <ndhist/detail/ndarray_storage.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

class ndhist
{
  public:


    ndhist(
        boost::numpy::ndarray const & shape
      , boost::python::list const & edges
      , boost::numpy::dtype const & dt
    )
      : bc_(ConstructBinContentStorage(shape, dt))
    {

        if(shape.get_size() != boost::python::len(edges))
        {
            throw ValueError(
                "The size of the shape array and the length of the edges list "
                "must be equal!");
        }
    }

    virtual ~ndhist() {}

    boost::numpy::ndarray
    GetBinContentArray();

    intptr_t
    GetAddr() const
    { return reinterpret_cast<intptr_t>(boost::python::object(*this).ptr()); }
  private:
    ndhist() {};

    /** Constructs a ndarray_storage object for the bin contents. The extra
     *  front and back capacity will be set to zero.
     */
    static
    detail::ndarray_storage
    ConstructBinContentStorage(bn::ndarray const & shape, bn::dtype const & dt);

    /** The bin contents.
     */
    detail::ndarray_storage bc_;

    /** The vector of the edges arrays.
     */
    std::vector<detail::ndarray_storage> edges_;
};


detail::ndarray_storage
ndhist::
ConstructBinContentStorage(bn::ndarray const & shape, bn::dtype const & dt)
{
    if(shape.get_nd() != 1)
    {
        throw ValueError(
            "The shape array must be 1-dimensional!");
    }

    std::vector<intptr_t> shape_vec = shape.as_vector<intptr_t>();

    // This function does not allow to specify extra front and back
    // capacity, so we set it to zero for all dimensions.
    std::vector<intptr_t> front_capacity(shape.get_size(), 0);
    std::vector<intptr_t> back_capacity(shape.get_size(), 0);

    return detail::ndarray_storage(shape_vec, front_capacity, back_capacity, dt);
}

}// namespace ndhist

#endif
