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
#include <ndhist/detail/bytearray.hpp>

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
 */
class ndarray_storage
{
  public:
    /**
     * @brief Calculates the data offset w.r.t. the bytearray_data_offset for
     *     the given data type, shape, front and back capacities.
     *     If given, the sub_item_byte_offset is added to the address.
     */
    static
    intptr_t
    calc_first_shape_element_data_offset(
        bn::dtype const & dt
      , std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , intptr_t const sub_item_byte_offset=0
    );

    /**
     * @brief Calculates the data strides based on the given dtype object, the
     *     given shape, front and back capacities.
     */
    static
    void
    calc_data_strides(
        std::vector<intptr_t> & strides
      , bn::dtype const & dt
      , std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
    );

    /**
     * @brief Constructs a ndarray wrapping the data contained in the given
     *     ndarray_storage object using the specified data type, shape, front
     *     and back capacities, and an optional sub item byte offset.
     */
    static
    bn::ndarray
    construct_ndarray(
        ndarray_storage const & storage
      , bn::dtype const & dt
      , std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , intptr_t const sub_item_byte_offset = 0
      , bp::object const * data_owner = NULL
      , bool set_owndata_flag = true
    );

    /**
     * @brief Default constructor.
     */
    ndarray_storage()
      : dt_(bn::dtype::get_builtin<void>())
      , bytearray_data_offset_(0)
    {}

    /**
     * @brief Constructs a new ndarray_storage with new (c-contiguous) allocated
     *     data with the specified data type, shape, front- and back capacities.
     */
    ndarray_storage(
        boost::numpy::dtype   const & dt
      , std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
    )
      : shape_(shape)
      , front_capacity_(front_capacity)
      , back_capacity_(back_capacity)
      , dt_(bn::dtype(dt))
      , bytearray_data_offset_(0)
      , bytearray_(create_bytearray(shape_, front_capacity_, back_capacity_, dt_.get_itemsize()))
    {
        data_strides_.resize(shape_.size());
        calc_data_strides(data_strides_, dt_, shape_, front_capacity_, back_capacity_);
    }

    /**
     * @brief Constructs a new ndarray_storage that defines a data view into the
     *     bytearray of an other ndarray_storage object.
     */
    ndarray_storage(
        ndarray_storage const & base
      , intptr_t const bytearray_data_offset
      , std::vector<intptr_t> const & data_shape
      , std::vector<intptr_t> const & data_strides
    )
      : shape_(data_shape)
      , front_capacity_(std::vector<intptr_t>(shape_.size(), 0))
      , back_capacity_(std::vector<intptr_t>(shape_.size(), 0))
      , dt_(base.get_dtype())
      , data_strides_(data_strides)
      , bytearray_data_offset_(bytearray_data_offset)
      , bytearray_(base.bytearray_)
    {}

    /**
     * @brief Changes the view on the bytearray. The vector arguments should
     *     provide the difference w.r.t. the old data view.
     */
    void
    change_view(
        std::vector<intptr_t> const & delta_shape
      , std::vector<intptr_t> const & delta_front_capacity
      , std::vector<intptr_t> const & delta_back_capacity
    );

    /**
     * @brief Sets the entire bytearray to zero.
     */
    void
    clear()
    {
        bytearray_->clear();
    }

    /**
     * @brief Creates a deep copy of this ndarray_storage object, i.e. the
     *     underlaying bytearray is also copied.
     */
    ndarray_storage
    deepcopy() const
    {
        ndarray_storage thecopy(*this);
        thecopy.bytearray_ = this->bytearray_->deepcopy();
        return thecopy;
    }

    /**
     * @brief Creates a shallow copy of this ndarray_storage object by using
     *     the copy constructor, i.e. the underlaying bytearray is not copied.
     */
    ndarray_storage
    shallowcopy() const
    {
        return ndarray_storage(*this);
    }

    /**
     * @brief Constructs a boost::numpy::ndarray object wrapping the data of
     *     this ndarray storage with the correct layout, i.e. offset and
     *     strides.
     *     If the field_idx is greater than 0, it is assumed, that the data
     *     storage was created with a structured dtype object and the correct
     *     byte offset will be calculated automatically to select the field
     *     having the given index.
     */
    bn::ndarray
    construct_ndarray(
        bn::dtype const &  dt
      , size_t             field_idx = 0
      , bp::object const * data_owner = NULL
      , bool               set_owndata_flag = true
    ) const;

    /**
     * @brief Creates a ndarray_storage object, that defines a view into the
     *     data of this ndarray_storage object.
     *     The bytearray_data_offset is supposed to be the offset w.r.t. the
     *     bytearray_data_offset of this ndarray_storage object.
     */
    inline
    ndarray_storage
    create_data_view(
        intptr_t const bytearray_data_offset
      , std::vector<intptr_t> const & data_shape
      , std::vector<intptr_t> const & data_strides
    ) const
    {
        return ndarray_storage(
            *this
          , bytearray_data_offset_ + bytearray_data_offset
          , data_shape
          , data_strides
        );
    }

    inline
    size_t
    get_nd() const
    {
        return shape_.size();
    }

    inline
    std::vector<intptr_t> const &
    get_front_capacity_vector() const
    {
        return front_capacity_;
    }

    inline
    std::vector<intptr_t> const &
    get_back_capacity_vector() const
    {
        return back_capacity_;
    }

    inline
    std::vector<intptr_t> const &
    get_shape_vector() const
    {
        return shape_;
    }

    inline
    bn::dtype const &
    get_dtype() const
    {
        return dt_;
    }

    inline
    intptr_t
    get_bytearray_data_offset() const
    {
        return bytearray_data_offset_;
    }

    inline
    intptr_t
    calc_first_shape_element_data_offset()
    {
        return calc_first_shape_element_data_offset(dt_, shape_, front_capacity_, back_capacity_, 0);
    }

    /**
     * @brief Returns the raw pointer to the beginning of the data junk used by
     *     this ndarray_storage object.
     */
    inline
    char *
    get_data() const
    {
        return bytearray_->get();
    }

    inline
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

    /**
     * @brief Extends the memory of this ndarray storage by at least the given
     *     number of elements for each axis. The f_n_elements_vec argument must
     *     hold the number of extra front elements (can be zero) for each axis.
     *     The b_n_elements_vec argument must hold the number of extra back
     *     elements (can be zero) for each axis. If all the axes have still
     *     enough capacity to hold the new elements, no reallocation of
     *     memory is performed. Otherwise a complete new junk of memory is
     *     allocated that can fit the extended array plus the specified extra
     *     font (max_fcap_vec) and back (max_bcap_vec) capacity.
     *     The data from the old array is copied to the new array.
     * @note: If this ndarray_storage does not own the bytearray, i.e. the data,
     *     this function will reallocate the data anyways and this
     *     ndarray_storage will own the data.
     */
    void
    extend_axes(
        std::vector<intptr_t> const & f_n_elements_vec
      , std::vector<intptr_t> const & b_n_elements_vec
      , std::vector<intptr_t> const & max_fcap_vec
      , std::vector<intptr_t> const & max_bcap_vec
    );

    /** The shape defines the number of dimensions and how many elements each
     *  dimension has.
     */
    std::vector<intptr_t> shape_;

    /** The additional front and back capacities define how many additional
     *  elements each dimension has before and after the shape elements.
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

    /** The data offset specifies the byte offset for this data view w.r.t.
     *  bytearray_->get(), i.e. start address of the byte data.
     */
    intptr_t bytearray_data_offset_;

    /** The shared pointer to the bytearray, that might be shared between
     *  different ndarray_storage objects.
     */
    boost::shared_ptr<bytearray> bytearray_;

  protected:
    /** Creates (i.e. allocates data) bytearray object for the given array
     *  layout. It returns a shared pointer to the new created bytearray object.
     */
    static
    boost::shared_ptr<bytearray>
    create_bytearray(
        std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , size_t const itemsize
    );
};

}// namespace detail
}// namespace ndhist

#endif // ! NDHIST_DETAIL_NDARRAY_STORAGE_H_INCLUDED
