#ifndef NDHIST_NDHIST_H_INCLUDED
#define NDHIST_NDHIST_H_INCLUDED 1

#include <stdint.h>

#include <vector>

#include <ndhist/detail/error.hpp>
#include <ndhist/dtype.hpp>
#include <ndhist/detail/NDDataArray.hpp>

namespace ndhist {

class ndhist
{
  public:
    ndhist(
          std::vector<uint32_t> nbins
        , std::vector<detail::NDDataArray> edges)
    {
        if(nbins.size() != edges.size())
        {
            throw error("Dimension ");
        }
    }
  private:
    ndhist() {};

    /// The bin contents.
    boost::shared_ptr<detail::NDDataArray> bc_ptr_;

    /// The vector of ed
    std::vector<detail::NDDataArray> edges_;
};

}// namespace ndhist

#endif
