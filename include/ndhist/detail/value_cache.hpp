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
    std::vector<intptr_t> relative_indices_;
    WeightValueType weight_;
};

struct ValueCacheBase
{
    intptr_t nd_;
    intptr_t capacity_;
    intptr_t size_;

    boost::function<bool (ValueCacheBase &, std::vector<intptr_t> const &, void *)>
        push_back_fct_;
    boost::function<void const * (ValueCacheBase const &, intptr_t)>
        get_entry_fct_;

    ValueCacheBase()
      : nd_(0)
      , capacity_(0)
      , size_(0)
    {}

    ValueCacheBase(
        intptr_t nd
      , intptr_t capacity
    )
      : nd_(nd)
      , capacity_(capacity)
      , size_(0)
    {}

    intptr_t
    get_capacity() const
    {
        return capacity_;
    }

    intptr_t
    get_size() const
    {
        return size_;
    }

    void
    clear()
    {
        size_ = 0;
    }

    bool
    push_back(std::vector<intptr_t> const & relative_indices, void * weight_ptr)
    {
        return push_back_fct_(*this, relative_indices, weight_ptr);
    }

    void const *
    get_entry(intptr_t idx)
    {
        return get_entry_fct_(*this, idx);
    }
};


template <typename WeightValueType>
struct ValueCache
  : ValueCacheBase
{
    typedef WeightValueType
            weight_value_type;

    typedef ValueCacheBase
            base_t;

    typedef ValueCache<weight_value_type>
            type;

    typedef cache_entry<WeightValueType>
            cache_entry_type;


    std::vector<cache_entry_type> stack_;

    ValueCache(
        intptr_t nd
      , intptr_t capacity
    )
      : base_t(nd, capacity)
    {
        // Initialize the stack.
        cache_entry_type entry;
        entry.relative_indices_.resize(nd_);
        stack_.resize(capacity_, entry);
        //std::cout << "ValueCache stack size = "<<stack_.size() <<std::endl;

        // Set function pointers.
        push_back_fct_ = &type::push_back;
        get_entry_fct_ = &type::get_entry;
    }

    /** Returns true, when the capacity is reached after adding the entry, and
     *  false otherwise.
     */
    static
    bool
    push_back(
        ValueCacheBase & value_cache_base
      , std::vector<intptr_t> const & relative_indices
      , void * weight_ptr
    )
    {
        type & value_cache = *static_cast<type *>(&value_cache_base);
        weight_value_type weight = *reinterpret_cast<weight_value_type *>(weight_ptr);

        std::cout << "ValueCache::push_back at "<< value_cache.size_ << std::endl<<std::flush;

        memcpy(&value_cache.stack_[value_cache.size_].relative_indices_[0], &relative_indices[0], relative_indices.size()*sizeof(intptr_t));
        value_cache.stack_[value_cache.size_].weight_ = weight;
        ++value_cache.size_;
        return (value_cache.size_ == value_cache.capacity_);
    }

    static
    void const *
    get_entry(
        ValueCacheBase const & value_cache_base
      , intptr_t idx
    )
    {
        type const & value_cache = *static_cast<type const *>(&value_cache_base);
        return &value_cache.stack_[idx];
    }
};

}// namespace detail
}// namespace ndhist

#endif // !NDHIST_DETAIL_VALUE_CACHE_HPP_INCLUDED
