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
#ifndef NDHIST_DETAIL_VALUE_CACHE_HPP_INCLUDED
#define NDHIST_DETAIL_VALUE_CACHE_HPP_INCLUDED 1

#include <cstring>
#include <vector>

#include <ndhist/limits.hpp>

namespace ndhist {
namespace detail {

template <typename WeightValueType>
struct cache_entry
{
    std::vector<intptr_t> relative_indices;
    WeightValueType weight;
};

struct ValueCacheBase
{
    ValueCacheBase(intptr_t nd, intptr_t capacity)
      : nd_(nd)
      , capacity_(capacity)
    {}

    intptr_t nd_;
    intptr_t capacity_;
};


template <typename WeightValueType>
struct ValueCache
  : ValueCacheBase
{
    typedef ValueCacheBase
            base_t;

    typedef WeightValueType
            weight_value_type;

    typedef cache_entry<WeightValueType>
            cache_entry_type;

    intptr_t size_;
    std::vector<cache_entry_type> stack_;

    ValueCache(intptr_t nd, intptr_t capacity)
      : base_t(nd, capacity)
      , size_(0)
    {
        // Initialize the stack.
        cache_entry_type entry;
        entry.relative_indices.resize(nd_);
        stack_.resize(capacity_, entry);
        //std::cout << "ValueCache stack size = "<<stack_.size() <<std::endl;
    }

    /** Returns true, when the capacity is reached after adding the entry, and
     *  false otherwise.
     */
    bool
    push_back(
        std::vector<intptr_t> const & relative_indices
      , weight_value_type weight
    )
    {
        std::cout << "ValueCache::push_back at "<< size_ << std::endl<<std::flush;

        memcpy(&stack_[size_].relative_indices[0], &relative_indices[0], relative_indices.size()*sizeof(intptr_t));
        stack_[size_].weight = weight;
        ++size_;
        return (size_ == base_t::capacity_);
    }

    void
    clear()
    {
        size_ = 0;
    }

    intptr_t
    get_size() const
    {
        return size_;
    }

    cache_entry_type const &
    get_entry(intptr_t idx) const
    {
        return stack_[idx];
    }
};

}// namespace detail
}// namespace ndhist

#endif // !NDHIST_DETAIL_VALUE_CACHE_HPP_INCLUDED
