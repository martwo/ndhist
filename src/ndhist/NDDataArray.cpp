#include <cstddef>
#include <cstdlib>

#include <ndhist/detail/NDDataArray.hpp>

//______________________________________________________________________________
bool
NDDataArray::
calloc(size_t capacity, size_t elsize)
{
    data_ = (char*)calloc(capacity, elsize);
    if(data == NULL)
    {
        // No memory could be allocated.
        return false;
    }
    byte_capacity_ = capacity*elsize;
    return true;
}

//______________________________________________________________________________
void
NDDataArray::
free()
{
    free(data_);
    byte_capacity_ = 0;
}

//______________________________________________________________________________
bool
NDDataArray::
initialize(dtype_desc const & dt, size_t capacity)
{
    dtype_desc_ = dt;
    return this->calloc(capacity, dt.GetSize());
}