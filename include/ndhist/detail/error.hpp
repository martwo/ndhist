#ifndef NDHIST_DETAIL_ERROR_HPP_INCLUDED
#define NDHIST_DETAIL_ERROR_HPP_INCLUDED 1

namespace ndhist {
namespace detail {

class error: public std::exception
{
  public:
    error(std::string msg)
      : msg_(msg)
    {}

    const char* what() const throw()
    {
        return msg_.c_str();
    }

    ~error() throw() {}

  protected:
    std::string msg_;
};

}// namespace detail
}// namespace ndhist

#endif
