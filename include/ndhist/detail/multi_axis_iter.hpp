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
#ifndef NDHIST_DETAIL_MULTI_AXIS_ITER_HPP_INCLUDED
#define NDHIST_DETAIL_MULTI_AXIS_ITER_HPP_INCLUDED 1

#include <cstring>
#include <vector>

#include <boost/assert.hpp>

#include <boost/numpy/iterators/flat_iterator.hpp>

namespace ndhist {
namespace detail {

/**
 * @brief Iterates over all array elements of the axes which are not selected
 *        through the *fixed_axes_indices* vector. The indices of these axes
 *        are fixed.
 *        The iteration range on the non-fixed iterated axes can be limited
 *        through the *iter_axes_range_min* and
 *        *iter_axes_range_max* vectors. The range is always [min, max).
 *        For each iterated value the ``dereference`` method can be called to
 *        get the current value.
 */
template <typename ValueTypeTraits>
struct multi_axis_iter
{
    typedef typename ValueTypeTraits::value_ref_type
            value_ref_type;

    multi_axis_iter(
        bn::ndarray const & arr
      , bn::detail::iter_operand_flags_t arr_access_flags = bn::detail::iter_operand::flags::READONLY::value
    )
      : is_end_point_(false)
      , iter_(bn::iterators::flat_iterator<ValueTypeTraits>(arr, arr_access_flags))
      , arr_shape_(arr.get_shape_vector())
    {
        int const nd = arr_shape_.size();
        fixed_axes_mask_.resize(nd);
        indices_.resize(nd);
        iter_strides_.resize(nd);

        // Create the iterstride vector to calculate the iter_index.
        iter_strides_[nd-1] = 1;
        for(intptr_t i=nd-2; i>=0; --i)
        {
            iter_strides_[i] = arr_shape_[i+1]*iter_strides_[i+1];
        }
    }

    /**
     * @brief Initializes the iteration over the given iteration axes ranges
     *        for the fixed axes indices given by *fixed_axes_indices*.
     *        The iteration range on the non-fixed iterated axes can be limited
     *        through the *iter_axes_range_min* and *iter_axes_range_max*
     *        vectors. The range is always [min, max). All vectors must be of
     *        length *nd* -- the dimensionality of the array.
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
            fixed_axes_mask_[i] = (fixed_axes_indices[i] != axis::FLAGS_FLOATING_INDEX);
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
                indices_[i] = fixed_axes_indices[i];
            }
        }

        p_ = last_iter_axis_;

        iter_.jump_to_iter_index(calc_detail_iter_index());
        iter_index_ = 0;

        is_end_point_ = false;
    }

    void
    init_full_iteration()
    {
        int const nd = arr_shape_.size();
        std::vector<intptr_t> fixed_axes_indices(nd, axis::FLAGS_FLOATING_INDEX);
        std::vector<intptr_t> iter_axes_range_min(nd, 0);

        init_iteration(fixed_axes_indices, iter_axes_range_min, arr_shape_);
    }

    intptr_t
    calc_detail_iter_index()
    {
        int const nd = arr_shape_.size();
        intptr_t iter_index = 0;
        for(intptr_t i=0; i<nd; ++i)
        {
            iter_index += indices_[i]*iter_strides_[i];
        }
        return iter_index;
    }

    std::vector<intptr_t> const &
    get_indices() const
    {
        return indices_;
    }

    typename ValueTypeTraits::value_ref_type
    dereference()
    {
        return iter_.dereference();
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
        while((indices_[p_] == iter_axes_range_max_[p_]-1) || fixed_axes_mask_[p_])
        {
            --p_;
        }
        BOOST_ASSERT(p_ >= 0);
        ++indices_[p_];
        while(p_ < last_iter_axis_)
        {
            ++p_;
            if(! fixed_axes_mask_[p_]) {
                indices_[p_] = iter_axes_range_min_[p_];
            }
        }

        iter_.jump_to_iter_index(calc_detail_iter_index());
        ++iter_index_;
    }

    bool is_end() const
    {
        return is_end_point_;
    }

    char *
    get_data()
    {
        return iter_.get_detail_iter().get_data(0);
    }

    bool is_end_point_;
    bn::iterators::flat_iterator<ValueTypeTraits> iter_;
    std::vector<intptr_t> const arr_shape_;
    std::vector<intptr_t> iter_strides_;
    // True if the iteration should not iterate over this axis.
    std::vector<bool> fixed_axes_mask_;
    // The subrange of the iterated axes.
    std::vector<intptr_t> iter_axes_range_min_;
    std::vector<intptr_t> iter_axes_range_max_;
    // The current multi indices of the iteration.
    std::vector<intptr_t> indices_;
    intptr_t last_iter_axis_;
    intptr_t n_iterations_;
    intptr_t p_;
    intptr_t iter_index_;
};

}// namespace detail
}// namespace ndhist

#endif // !NDHIST_DETAIL_MULTI_AXIS_ITER_HPP_INCLUDED
