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
 *
 */
class ndarray_storage
  : public boost::noncopyable
{
  public:
    /**
     * @brief Calculates the offset of the data pointer needed for a ndarray
     *     wrapping a ndarray storage. If given, the sub_item_byte_offset is
     *     added to the address.
     */
    static
    intptr_t
    calc_data_offset(
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
    );

    /**
     * @brief Constructs a new ndarray_storage with new allocated data with the
     *     specified shape and data type.
     */
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
      , data_offset_(calc_data_offset(dt_, shape_, front_capacity_, back_capacity_, 0))
      , bytearray_(create_bytearray(shape_, front_capacity_, back_capacity_, dt_.get_itemsize()))
    {
        data_strides_.resize(shape_.size());
        calc_data_strides(data_strides_, dt_, shape_, front_capacity_, back_capacity_);
    }

    /**
     * @brief Constructs a new ndarray_storage that shares data with an other
     *     ndarray_storage, i.e. the new created ndarray_storage is just a view
     *     into the data that is owned by the given ndarray_storage object (or
     *     its data owner).
     */
    ndarray_storage(
        std::vector<intptr_t> const & shape
      , std::vector<intptr_t> const & front_capacity
      , std::vector<intptr_t> const & back_capacity
      , boost::numpy::dtype   const & dt
      , boost::shared_ptr<ndarray_storage> const & parent
    )
      : shape_(shape)
      , front_capacity_(front_capacity)
      , back_capacity_(back_capacity)
      , dt_(dt)
      , data_offset_(calc_data_offset(dt_, shape_, front_capacity_, back_capacity_, 0))
      , bytearray_(parent->bytearray_)
      , bytearray_owner_(parent->bytearray_owner_.get() == NULL ? parent : parent->bytearray_owner_)
    {
        // Check if the dimensionality of the view is not greater than one of
        // the parent.
        size_t const nd = get_nd();
        if(nd > bytearray_owner_->get_nd())
        {
            std::stringstream ss;
            ss << "The dimensionality of the child array layout ("<< nd
               <<") must be "
               << "smaller or equal than the dimensionality of the parent "
               << "array layout ("<< bytearray_owner_->get_nd() <<")!";
            throw ValueError(ss.str());
        }

        // Check if the view-shape is smaller or equal than the one of the
        // parent.
        for(size_t i=0; i<nd; ++i)
        {
            if((front_capacity_[i] + shape_[i] + back_capacity_[i]) >
               (bytearray_owner_->front_capacity_[i] + bytearray_owner_->shape_[i] + bytearray_owner_->back_capacity_[i])
              )
            {
                std::stringstream ss;
                ss << "The front capacity + shape + back capacity of the child "
                   << "array layout must be smaller or equal than that of the "
                   << "parent array layout for axis "<<i<<"!";
                throw ValueError(ss.str());
            }
        }

        // Check if the item size of the view is not greater than the one of
        // the parent.
        if(dt_.get_itemsize() > bytearray_owner_->dt_.get_itemsize())
        {
            std::stringstream ss;
            ss << "The item size of the view data type must not be greater "
               << "than the one of data type of the parent array layout ("
               << bytearray_owner_->dt_.get_itemsize() <<")!";
            throw TypeError(ss.str());
        }

        data_strides_.resize(shape_.size());
        calc_data_strides(data_strides_, dt_, shape_, front_capacity_, back_capacity_);
    }

    size_t get_nd() const { return shape_.size(); }

    /** Constructs a boost::numpy::ndarray object wrapping this ndarray storage
     *  with the correct layout, i.e. offset and strides. If the field_idx
     *  is greater than 0, it is assumed, that the data storage was created with
     *  a structured dtype object and the correct byte offset will be calculated
     *  automatically to select the field having the given index.
     */
    bn::ndarray
    construct_ndarray(
        bn::dtype const &  dt
      , size_t             field_idx = 0
      , bp::object const * data_owner = NULL
    );

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
    get_data_offset() const
    {
        return data_offset_;
    }

    /**
     * @brief Returns the raw pointer to the beginning of the data junk used by
     *     this ndarray_storage object.
     */
    inline
    char *
    get_data()
    {
        return bytearray_->data_;
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

    /** The data address offset for the first sub item of the structured data
     *  type using the set shape, front and back capacities.
     */
    intptr_t data_offset_;

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
