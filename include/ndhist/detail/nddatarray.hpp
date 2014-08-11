#ifndef NDHIST_DETAIL_NDDATAARRAY_H_INCLUDED
#define NDHIST_DETAIL_NDDATAARRAY_H_INCLUDED 1

#include <boost/numpy/dtype.hpp>

#include <ndhist/detail/error.hpp>

namespace bn = boost::numpy;

namespace ndhist {
namespace detail {

/**
 * The NDDataArray class provides a generic contigious byte data array
 * mapping into a multi-dimensional space.
 */
class NDDataArray
{
  public:
    NDDataArray(
        std::vector<intptr_t> shape
      , std::vector<intptr_t> add_front_capacity
      , std::vector<intptr_t> add_back_capacity
      , bn::dtype const & dt
    )
      : shape_(shape)
      , add_front_capacity_(add_front_capacity)
      , add_back_capacity_(add_back_capacity)
      , dt_(dt)
      , data_(NULL)
    {
        const size_t nd = shape_.size();
        if(! (add_front_capacity_.size() == nd &&
              add_back_capacity_.size()  == nd)
          )
        {
            throw error(
                "The lengthes of shape, add_front_capacity and "
                "add_back_capacity must be equal!");
        }
        if(nd == 0)
        {
            throw error(
                "The array must be at least 1-dimensional, i.e. "
                "len(shape) > 0!");
        }
        size_t capacity = 1;
        for(size_t i=0; i<nd; ++i)
        {
            const size_t cap_i = add_front_capacity_[i] + shape_[i] + add_back_capacity_[i];
            capacity *= cap_i;
        }
        if(! capacity > 0)
        {
            throw error(
                "The capacity is less or equal 0!");
        }
        Calloc(capacity, dt_.GetSize());
    }

    virtual
    ~NDDataArray()
    {
        if(data_)
        {
            Free();
        }
    }

  protected:
    /**
     * Allocates capacity*elsize number of bytes of new memory, initialized to
     * zero. It returns "true" after success and "false" otherwise. It uses the
     * calloc C function, which is faster than malloc + memset for large chunks
     * of memory allocation, due to OS specific memory management.
     * (cf. http://stackoverflow.com/questions/2688466/why-mallocmemset-is-slower-than-calloc)
     */
    void Calloc(size_t capacity, size_t elsize);

    /**
     * Calls free on data_ and sets data_ to NULL.
     */
    void Free();


  private:
    /** The shape defines the number of dimensions and how many elements each
     *  dimension has.
     */
    std::vector<intptr_t> shape_;

    /** The additional front and back capacities define how many additional
     *  elements each dimension has.
     */
    std::vector<intptr_t> add_front_capacity_;
    std::vector<intptr_t> add_back_capacity_;

    /** The numpy data type object, defining the element size in bytes.
     */
    bn::dtype dt_;

    /** The actual data storage.
     */
    char* data_;
};

}// namespace detail
}// namespace ndhist

#endif
