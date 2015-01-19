/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <ndhist@martin-wolf.org>
 *
 * This file is distributed under the BSD 2-Clause Open Source License
 * (See LICENSE file).
 *
 */
#ifndef NDHIST_DETAIL_NDARRAY_STORAGE_H_INCLUDED
#define NDHIST_DETAIL_NDARRAY_STORAGE_H_INCLUDED 1

#include <iostream>
#include <vector>

#include <boost/python.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>

#include <ndhist/error.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {
namespace detail {

/**
 * The ndarray_storage class provides storage management functionalities for
 * a ndarray object through a generic contigious byte array. This byte array
 * can be accessed via a ndarray object which can be build around this storage.
 * The storage can be bigger than what the ndarray accesses. This allows to add
 * additional (hidden) capacity to arrays, which can be used for growing the
 * ndarray without memory re-allocation.
 *
 */
class ndarray_storage
  : public boost::noncopyable
{
  public:
    ndarray_storage(
        std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , boost::numpy::dtype   const & dt
    )
      : shape_(shape)
      , front_capacity_(front_capacity)
      , back_capacity_(back_capacity)
      , dt_(dt)
      , data_(NULL)
    {
        data_strides_.resize(shape_.size());
        calc_data_strides(data_strides_);
        data_ = create_array_data(shape_, front_capacity_, back_capacity_, dt_.get_itemsize());
    }

    virtual
    ~ndarray_storage()
    {
        //std::cout << "Destructing ndarray_storage" << std::endl;
        if(data_)
        {
            free_data(data_);
            data_ = NULL;
        }
    }

    int get_nd() const { return shape_.size(); }

    /** Calculates the offset of the data pointer needed for a ndarray wrapping
     *  this ndarray storage.
     */
    intptr_t CalcDataOffset(size_t sub_item_byte_offset) const;

    /** Calculates the data strides for the dtype object of this ndarray
     *  storage.
     */
    void calc_data_strides(std::vector<intptr_t> & stides) const;

    /** Constructs a boost::numpy::ndarray object wrapping this ndarray storage
     *  with the correct layout, i.e. offset and strides. If the field_idx
     *  is greater than 0, it is assumed, that the data storage was created with
     *  a structured dtype object and the correct byte offset will be calculated
     *  automatically to select the field having the given index.
     */
    bn::ndarray
    ConstructNDArray(
        bn::dtype const &  dt
      , size_t             field_idx = 0
      , bp::object const * data_owner = NULL
    );

    std::vector<intptr_t> const &
    get_front_capacity_vector() const
    {
        return front_capacity_;
    }

    std::vector<intptr_t> const &
    get_back_capacity_vector() const
    {
        return back_capacity_;
    }

    std::vector<intptr_t> const &
    get_shape_vector() const
    {
        return shape_;
    }

    bn::dtype const &
    get_dtype() const
    {
        return dt_;
    }

    std::vector<intptr_t> const &
    get_data_strides_vector() const
    {
        return data_strides_;
    }

    /**
     * @brief Copies the data of the given source array into this storage. The
     *     shape of the source array for each axis must not be greater than the
     *     shape of this storage. In case the shape of the source array is
     *     smaller for some axes, the shift (offset) of each axis within this
     *     storage can be specified.
     */
    void
    copy_from(
        bn::ndarray const & src_arr
      , std::vector<intptr_t> const & shape_offset_vec
    );

    /** Extends the memory of this ndarray storage by at least the given number
     *  of elements for each axis. The f_n_elements_vec argument must hold the
     *  number of extra front elements (can be zero) for each axis. The
     *  b_n_elements_vec argument must hold the number of extra back elements
     *  (can be zero) for each axis. If all the axes
     *  have still enough capacity to hold the new elements, no reallocation of
     *  memory is performed. Otherwise a complete new junk of memory is
     *  allocated that can fit the extended array plus the specified extra font
     *  (max_fcap_vec) and back (max_bcap_vec) capacity. The data from the old
     *  array is copied to the new array.
     */
    void
    extend_axes(
        std::vector<intptr_t> const & f_n_elements_vec
      , std::vector<intptr_t> const & b_n_elements_vec
      , std::vector<intptr_t> const & max_fcap_vec
      , std::vector<intptr_t> const & max_bcap_vec
      , bp::object const * data_owner
    );

  protected:
    /** Creates (i.e. allocates) data for the given array layout. It returns
     *  the pointer to the new allocated memory.
     */
    static
    char *
    create_array_data(
        std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , size_t itemsize
    );

  private:
    /** The shape defines the number of dimensions and how many elements each
     *  dimension has.
     */
    std::vector<intptr_t> shape_;

    /** The additional front and back capacities define how many additional
     *  elements each dimension has.
     */
    std::vector<intptr_t> front_capacity_;
    std::vector<intptr_t> back_capacity_;

    /** The numpy data type object, defining the element size in bytes.
     */
    bn::dtype dt_;

    /** The vector holding the strides information for the data type as given
     *  by the dt_ dtype object.
     */
    std::vector<intptr_t> data_strides_;

  public:
    /** The shared pointer to the bytearray, that might be shared between
     *  different ndarray_storage objects.
     */
    boost::shared_ptr<bytearray> bytearray_;

    /** The owner of the bytearray. If that is non-NULL, this ndarray_storage
     *  object does not own the data and is not allowed to change it. In case
     *  changes (e.g. the shape) are required, the bytearray needs to be copied
     *  before.
     */
    boost::shared_ptr<ndarray_storage> bytearray_owner_;
};

}// namespace detail
}// namespace ndhist

#endif // ! NDHIST_DETAIL_NDARRAY_STORAGE_H_INCLUDED
