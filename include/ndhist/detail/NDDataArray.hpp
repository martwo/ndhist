#ifndef NDHIST_DETAIL_NDDATAARRAY_H_INCLUDED
#define NDHIST_DETAIL_NDDATAARRAY_H_INCLUDED 1

#include <ndhist/dtype.hpp>

/**
 * The NDDataArray class provides a generic contigious byte data array
 * mapping into a multi-dimensional space.
 */
class NDDataArray
{
  public:
    NDDataArray()
      : byte_capacity_(0)
      , data_(NULL)
    {}

    virtual ~NDDataArray()
    {
        if(byte_capacity_ > 0)
        {
            this->free();
        }
    }

    /**
     * Allocates capacity*elsize number of bytes of new memory, initialized to
     * zero. It returns "true" after success and "false" otherwise. It uses the
     * calloc C function, which is faster than malloc + memset for large chunks
     * of memory allocation, due to OS specific memory management.
     * (cf. http://stackoverflow.com/questions/2688466/why-mallocmemset-is-slower-than-calloc)
     */
    bool calloc(size_t capacity, size_t elsize);
    void free();

    /**
     * Initializes the NDDataArray with a given data type and a given capacity
     * of data type elements. It returns "true" after success and "false"
     * otherwise.
     */
    bool initialize(dtype_desc const & dt, size_t capacity);

  protected:
    size_t byte_capacity_;
    char* data_;
    dtype_desc dtype_desc_;
};

#endif
