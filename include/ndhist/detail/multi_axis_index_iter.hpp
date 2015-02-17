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
#ifndef NDHIST_DETAIL_FULL_MULTI_AXIS_INDEX_ITER_HPP_INCLUDED
#define NDHIST_DETAIL_FULL_MULTI_AXIS_INDEX_ITER_HPP_INCLUDED 1

#include <vector>

#include <boost/assert.hpp>

#include <ndhist/detail/axis.hpp>

namespace ndhist {
namespace detail {

/**
 * @brief The full_multi_axis_index_iter struct provides a n-dimensional index
 *        iterator over the full range of each axis. Each axis can be fixed to
 *        certain value if needed.
 */
struct full_multi_axis_index_iter
{
    full_multi_axis_index_iter(
        std::vector<intptr_t> const & arr_shape
    )
      : is_end_point_(false)
      , arr_shape_(arr_shape)
    {
        uintptr_t const nd = arr_shape_.size();
        fixed_axes_mask_.resize(nd);
        indices_.resize(nd);
    }

    /**
     * @brief Initializes the iteration over the array (including OOR bins)
     *        for the given fixed axes indices (*fixed_axes_indices*).
     *        All vectors must be of
     *        length *nd* -- the dimensionality of the array.
     */
    void
    init_iteration(std::vector<intptr_t> const & fixed_axes_indices)
    {
        uintptr_t const nd = arr_shape_.size();
        indices_.assign(nd, axis::OOR_UNDERFLOW);
        n_iterations_ = 1;
        last_iter_axis_ = -1;
        for(uintptr_t i=0; i<nd; ++i)
        {
            fixed_axes_mask_[i] = (fixed_axes_indices[i] != axis::FLAGS_FLOATING_INDEX);
            if(! fixed_axes_mask_[i])
            {
                last_iter_axis_ = i;
                n_iterations_ *= arr_shape_[i]+2;
            }
            else
            {
                indices_[i] = fixed_axes_indices[i];
            }
        }

        if(last_iter_axis_ == -1)
        {
            throw AssertionError("No iteration axes specified for oor multi-axis iteration!");
        }

        p_ = last_iter_axis_;
        iter_index_ = 0;
        is_end_point_ = false;
    }

    std::vector<intptr_t> const &
    get_indices() const
    {
        return indices_;
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
        // fixed axes).
        while((indices_[p_] == axis::OOR_OVERFLOW) || fixed_axes_mask_[p_])
        {
            --p_;
        }
        BOOST_ASSERT(p_ >= 0);
        if(indices_[p_] == arr_shape_[p_]-1) {
            indices_[p_] = axis::OOR_OVERFLOW;
        }
        else {
            ++indices_[p_];
        }
        while(p_ < last_iter_axis_)
        {
            ++p_;
            if(! fixed_axes_mask_[p_]) {
                indices_[p_] = axis::OOR_UNDERFLOW;
            }
        }

        ++iter_index_;
    }

    bool is_end() const
    {
        return is_end_point_;
    }

    bool is_oor_bin() const
    {
        uintptr_t const nd = arr_shape_.size();
        for(uintptr_t i=0; i<nd; ++i)
        {
            if((indices_[i] == axis::OOR_UNDERFLOW) ||
               (indices_[i] == axis::OOR_OVERFLOW))
            {
                return true;
            }
        }
        return false;
    }

    uintptr_t
    get_oor_array_index() const
    {
        uintptr_t idx = 0;
        uintptr_t const nd = arr_shape_.size();
        for(uintptr_t i=0; i<nd; ++i)
        {
            if(! ((indices_[i] == axis::OOR_UNDERFLOW) || (indices_[i] == axis::OOR_OVERFLOW)))
            {
                // Mark the axis i as available.
                idx |= (1<<i);
            }
        }
        return idx;
    }

    bool is_end_point_;
    std::vector<intptr_t> const & arr_shape_;
    std::vector<bool> fixed_axes_mask_;
    std::vector<intptr_t> indices_;
    intptr_t last_iter_axis_;
    intptr_t n_iterations_;
    intptr_t p_;
    intptr_t iter_index_;
};

}// namespace detail
}// namespace ndhist

#endif  // !NDHIST_DETAIL_FULL_MULTI_AXIS_INDEX_ITER_HPP_INCLUDED
