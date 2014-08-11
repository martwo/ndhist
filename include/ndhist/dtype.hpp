
#ifndef NDHIST_DTYPE_H_INCLUDED
#define NDHIST_DTYPE_H_INCLUDED 1

#include <stdint.h>

namespace ndhist {

/**
 * The dtype_desc class provides a base class for all data type classes
 * usable with ndhist. It stores the size of the particular data type in bytes.
 */
class dtype_desc
{
  public:
    dtype_desc()
      : size_(0)
    {}

    dtype_desc(size_t size)
      : size_(size)
    {}

    inline
    size_t GetSize() const { return size_; }

  protected:
    /// The size of the data type in bytes.
    size_t size_;
};

template <CPPType>
class dtype
  : public dtype_desc
{
  public:
    typedef CPPType cpp_type;

    dtype()
      : dtype_desc(sizeof(cpp_type))
    {}
};

class float64
  : public dtype<double>
{
  public:
    typedef dtype<double> base;
    typedef typename base::cpp_type cpp_type;

    float64()
      : base()
    {}
};

class uint64
  : public dtype<uint64_t>
{
  public:
    typedef dtype<uint64_t> base;
    typedef typename base::cpp_type cpp_type;

    uint64()
      : base()
    {}
};

class dtype_value
{
  public:
    dtype_value(dtype_desc const & dt)
      : dtype_desc_(dt)
      , data_(NULL)
    {}

    template <typename T>
    bool
    set(T value)
    {
        size_t sizeT = sizeof(T);
        // Check if T is compatible with the given dtype_desc.
        dtype<T> dt;
        if(dt.GetSize() != sizeT)
        {
            return false;
        }

        // Allocate memory, if not done so alreay before.
        if(data_ == NULL)
        {
            data_ = malloc(sizeT);
            if(data_ == NULL)
            {
                return false;
            }
        }

        memcpy(data_, &value, sizeT);
        return true;
    }

    template <typename T>
    T&
    get()
    {

    }
  protected:
    char* data_;
    dtype_desc dtype_desc_;
};

}// namespace ndhist

#endif
