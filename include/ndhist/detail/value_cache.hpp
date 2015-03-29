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

#include <boost/function.hpp>

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

    boost::function<boost::shared_ptr<ValueCacheBase> (ValueCacheBase const &)>
        deepcopy_fct_;

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

    boost::shared_ptr<ValueCacheBase>
    deepcopy() const
    {
        return deepcopy_fct_(*this);
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
        deepcopy_fct_ = &type::deepcopy;
    }

    static
    boost::shared_ptr<ValueCacheBase>
    deepcopy(ValueCacheBase const & value_cache_base)
    {
        type const & value_cache = *static_cast<type const *>(&value_cache_base);
        boost::shared_ptr<type> thecopy(new type(value_cache));
        return thecopy;
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

        memcpy(&stack_[size_].relative_indices_[0], &relative_indices[0], relative_indices.size()*sizeof(intptr_t));
        stack_[size_].weight_ = weight;
        ++size_;
        return (size_ == capacity_);
    }

    cache_entry_type const &
    get_entry(intptr_t idx)
    {
        return stack_[idx];
    }
};

}// namespace detail
}// namespace ndhist

#endif // !NDHIST_DETAIL_VALUE_CACHE_HPP_INCLUDED
