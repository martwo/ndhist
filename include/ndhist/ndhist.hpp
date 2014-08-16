#ifndef NDHIST_NDHIST_H_INCLUDED
#define NDHIST_NDHIST_H_INCLUDED 1

#include <stdint.h>

#include <vector>

#include <boost/shared_ptr.hpp>

#include <boost/numpy/ndarray.hpp>

#include <ndhist/detail/error.hpp>
#include <ndhist/detail/nddatarray.hpp>

namespace bn = boost::numpy;

namespace ndhist {

class ndhist
{
  public:
    ndhist(
          std::vector<uint32_t> nbins
        , std::vector<bn::ndarray> edges)
    {
        if(nbins.size() != edges.size())
        {
            throw error(
                "The lenghts of the nbins and edges arrays must be equal!");
        }
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
