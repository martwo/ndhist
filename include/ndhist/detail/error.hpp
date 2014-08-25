#ifndef NDHIST_DETAIL_ERROR_HPP_INCLUDED
#define NDHIST_DETAIL_ERROR_HPP_INCLUDED 1

namespace ndhist {
namespace detail {

struct AssertionErrorType {};
struct IndexErrorType     {};
struct MemoryErrorType    {};
struct ValueErrorType     {};

template<class ErrType>
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

#endif // ! NDHIST_DETAIL_ERROR_HPP_INCLUDED
