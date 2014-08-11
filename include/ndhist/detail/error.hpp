#ifndef NDHIST_DETAIL_ERROR_HPP_INCLUDED
#define NDHIST_DETAIL_ERROR_HPP_INCLUDED 1

namespace ndhist {
namespace detail {

class error: public std::exception
{
    error(std::string msg)
      : msg_(msg)

    virtual
    const char* what() const
    {
        return msg_.c_str();
    }

    std::string msg_;
};

}// namespace detail
}// namespace ndhist

#endif
