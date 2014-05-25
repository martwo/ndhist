
#ifndef NDHIST_DTYPE_H_INCLUDED
#define NDHIST_DTYPE_H_INCLUDED 1

#include <stdint.h>

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

#endif
