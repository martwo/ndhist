/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <ndhist@martin-wolf.org>
 *
 * This file is distributed under the BSD 2-Clause Open Source License
 * (See LICENSE file).
 *
 */
#ifndef NDHIST_DETAIL_MULTI_AXES_ITER_HPP_INCLUDED
#define NDHIST_DETAIL_MULTI_AXES_ITER_HPP_INCLUDED 1

#include <boost/assert.hpp>

namespace ndhist {
namespace detail {

/**
 * @brief Iterates over all array elements of the axes which are not selected
 *        through the *fixed_axes_range_min* vector. The indices of these axes are fixed
 *        and
 *        specified via the *axes_index* vector. The iteration range on the
 *        non-fixed iterated axes can be limited through the *axes_range_min*
 *        and *axes_range_max* vectors. The range is always [min, max).
 *        For each iterated value the callback function is called with the
 *        memory address of that value.
 */
template <typename ValueType>
struct multi_axis_iter
{
    multi_axis_iter(
        bn::ndarray & arr
      , bn::detail::iter_operand_flags_t arr_access_flags = bn::detail::iter_operand::flags::READONLY::value
    )
      : is_end_point_(false)
      , iter_(bn::flat_iterator<ValueType>(arr, arr_access_flags))
      , arr_shape_(iter_.get_detail_iter.get_operand(0).get_shape_vector())
    {
        int const nd = arr_shape_.size();
        fixed_axes_mask_.resize(nd);
        indices_.resize(nd);
        iter_strides_.resize(nd);

        // Create the iterstride vector to calculate the iter_index.
        iter_strides_[nd-1] = 1;
        for(intptr_t i=nd-2; i>=0; --i)
        {
            iter_strides[i] = arr_shape_[i+1]*iter_strides[i+1];
        }
    }

    /**
     * @brief Initializes the iteration over the given iteration axes ranges
     *        for the fixed axes indices given by *fixed_axes_indices*.
     *        The iteration range on the non-fixed iterated axes can be limited
     *        through the *iter_axes_range_min* and *iter_axes_range_max*
     *        vectors. The range is always [min, max). All vectors must be of
     *        length *nd* - the dimensionality of the array.
     */
    void
    init_iteration(
        std::vector<intptr_t> const & fixed_axes_indices
      , std::vector<intptr_t> const & iter_axes_range_min
      , std::vector<intptr_t> const & iter_axes_range_max
    )
    {
        int const nd = arr_shape_.size();

        iter_axes_range_min_ = iter_axes_range_min;
        iter_axes_range_max_ = iter_axes_range_max;

        memset(&indices_[0], 0, nd*sizeof(intptr_t));

        n_iterations_ = 1;
        last_iter_axis_ = -1;
        for(intptr_t i=0; i<nd; ++i)
        {
            fixed_axes_mask_[i] = (fixed_axes_indices[i] >= 0);
            if(! fixed_axes_mask_[i])
            {
                if(iter_axes_range_min_[i] < 0) {
                    iter_axes_range_min_[i] = 0;
                }
                if(iter_axes_range_max_[i] < 0) {
                    iter_axes_range_max_[i] = arr_shape_[i];
                }
                indices_[i] = iter_axes_range_min_[i];
                last_iter_axis_ = i;
                n_iterations_ *= iter_axes_range_max_[i] - iter_axes_range_min_[i];
            }
            else
            {
                indices_[i] = fixed_axes_index[i];
            }
        }

        if(last_iter_axis_ == -1)
        {
            throw AssertionError("No iteration axes specified for multi-axes iteration!");
        }
        p_ = last_iter_axis_;

        // Calculate the iter index of the first value addressed by *indices_*.
        _calc_iter_index();

        iter_.jump_to_iter_index(iter_index_);

        is_end_point_ = false;
    }

    void
    _calc_iter_index()
    {
        int const nd = arr_shape_.size();
        iter_index_ = 0;
        for(intptr_t i=0; i<nd; ++i)
        {
            iter_index_ += indices_[i]*iter_strides_[i];
        }
    }

    char * get_data() const
    {
        return iter.get_detail_iter().get_data(0);
    }

    void increment()
    {
        // Increment the indices values for the next value.
        if(iter_index_ == n_iterations_-1)
        {
            is_end_point_ = true;
            return;
        }

        // Move the index pointer to the next non-fixed outer-axis if the index
        // of the current axis has reached its maximum. Then increase
        // the index and reset all indices to the right of this
        // increased index to zero. After this operation, the index
        // pointer points to the inner-most axis (excluding the
        // iteration axis).
        while((indices_[p_] == iter_axes_range_max_[p_]-1) || fixed_axes_mask_[p_])
        {
            --p_;
        }
        BOOST_ASSERT(p >= 0);
        ++indices_[p_];
        while(p_ < last_iter_axis_)
        {
            ++p_;
            if(! fixed_axes_mask_[p_]) {
                indices_[p_] = iter_axes_range_min_[p_];
            }
        }

        _calc_iter_index();

        iter_.jump_to_iter_index(iter_index_);
    }

    bool is_end() const
    {
        return is_end_point_;
    }

    bool is_end_point_;
    bn::flat_iterator<ValueType> iter_;
    std::vector<intptr_t> const arr_shape_;
    std::vector<intptr_t> iter_strides_;
    // True if the iteration should not iterate over this axis.
    std::vector<bool> fixed_axes_mask_;
    // The subrange of the iterated axes.
    std::vector<intptr_t> iter_axes_range_min_;
    std::vector<intptr_t> iter_axes_range_max_;
    std::vector<intptr_t> indices_;
    intptr_t last_axis_;
    intptr_t n_iterations_;
    intptr_t p_;
    intptr_t iter_index_;
};

}// namespace detail
}// namespace ndhist

#endif // !NDHIST_DETAIL_MULTI_AXES_ITER_HPP_INCLUDED
