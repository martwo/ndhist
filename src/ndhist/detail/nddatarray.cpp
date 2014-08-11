#include <cstddef>
#include <cstdlib>

#include <ndhist/detail/NDDataArray.hpp>

//______________________________________________________________________________
void
NDDataArray::
Calloc(size_t capacity, size_t elsize)
{
    data_ = (char*)calloc(capacity, elsize);
    if(data == NULL)
    {
        throw error("Unable to allocate memory!");
    }
    return true;
}

//______________________________________________________________________________
void
NDDataArray::
Free()
{
    free(data_);
    data_ = NULL;
}
