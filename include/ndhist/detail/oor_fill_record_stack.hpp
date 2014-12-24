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
#ifndef NDHIST_DETAIL_OOR_FILL_RECORD_STACK_HPP_INCLUDED
#define NDHIST_DETAIL_OOR_FILL_RECORD_STACK_HPP_INCLUDED 1

#include <ndhist/limits.hpp>

namespace ndhist {
namespace detail {

template <typename BCValueType>
struct oor_fill_record
{
    bool is_oor;
    uintptr_t oor_arr_idx;
    std::vector<intptr_t> oor_arr_noor_relative_indices;
    std::vector<intptr_t> oor_arr_noor_axes_indices;
    uintptr_t oor_arr_noor_relative_indices_size;
    std::vector<intptr_t> oor_arr_oor_relative_indices;
    std::vector<intptr_t> oor_arr_oor_axes_indices;
    uintptr_t oor_arr_oor_relative_indices_size;
    std::vector<intptr_t> relative_indices;
    BCValueType weight;
};

struct OORFillRecordStackBase
{
    OORFillRecordStackBase(intptr_t nd, intptr_t capacity)
      : nd_(nd)
      , capacity_(capacity)
    {
        nd_mem0_.resize(nd);
    }

    intptr_t nd_;
    intptr_t capacity_;

    // Memory that can be used as temporary memory of size ND.
    std::vector<intptr_t> nd_mem0_;
};


template <typename BCValueType>
class OORFillRecordStack
  : public OORFillRecordStackBase
{
  public:
    typedef OORFillRecordStackBase
            base_t;
    typedef BCValueType
            bc_value_type;
    typedef oor_fill_record<BCValueType>
            oor_fill_record_type;

    OORFillRecordStack(intptr_t nd, intptr_t capacity)
      : base_t(nd, capacity)
      , size_(0)
    {
        // Initialize the stack.
        oor_fill_record_type rec;
        rec.is_oor = false;
        rec.oor_arr_idx = 0;
        rec.oor_arr_noor_relative_indices.resize(nd);
        rec.oor_arr_noor_axes_indices.resize(nd);
        rec.oor_arr_noor_relative_indices_size = 0;
        rec.oor_arr_oor_relative_indices.resize(nd);
        rec.oor_arr_oor_axes_indices.resize(nd);
        rec.oor_arr_oor_relative_indices_size = 0;
        rec.relative_indices.resize(nd);
        stack_.resize(base_t::capacity_, rec);
        std::cout << "OORFillRecordStack size = "<<stack_.size() <<std::endl;
    }

    /** Returns true, when the capacity is reached after adding the record, and
     *  false otherwise.
     */
    bool
    push_back(
        bool is_oor
      , uintptr_t oor_arr_idx
      , std::vector<intptr_t> const & oor_arr_noor_relative_indices
      , std::vector<intptr_t> const & oor_arr_noor_axes_indices
      , uintptr_t oor_arr_noor_relative_indices_size
      , std::vector<intptr_t> const & oor_arr_oor_relative_indices
      , std::vector<intptr_t> const & oor_arr_oor_axes_indices
      , uintptr_t oor_arr_oor_relative_indices_size
      , std::vector<intptr_t> const & relative_indices
      , BCValueType weight
    )
    {
        std::cout << "OORFillRecordStack::push_back at "<< size_ << std::endl<<std::flush;
        stack_[size_].is_oor = is_oor;
        if(is_oor)
        {
            stack_[size_].oor_arr_idx = oor_arr_idx;
            memcpy(&stack_[size_].oor_arr_noor_relative_indices[0], &oor_arr_noor_relative_indices[0], oor_arr_noor_relative_indices_size*sizeof(intptr_t));
            memcpy(&stack_[size_].oor_arr_noor_axes_indices[0], &oor_arr_noor_axes_indices[0], oor_arr_noor_relative_indices_size*sizeof(intptr_t));
            stack_[size_].oor_arr_noor_relative_indices_size = oor_arr_noor_relative_indices_size;
            memcpy(&stack_[size_].oor_arr_oor_relative_indices[0], &oor_arr_oor_relative_indices[0], oor_arr_oor_relative_indices_size*sizeof(intptr_t));
            memcpy(&stack_[size_].oor_arr_oor_axes_indices[0], &oor_arr_oor_axes_indices[0], oor_arr_oor_relative_indices_size*sizeof(intptr_t));
            stack_[size_].oor_arr_oor_relative_indices_size = oor_arr_oor_relative_indices_size;
        }
        else
        {
            memcpy(&stack_[size_].relative_indices[0], &relative_indices[0], relative_indices.size()*sizeof(intptr_t));
        }
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

    oor_fill_record_type const &
    get_record(intptr_t idx) const
    {
        return stack_[idx];
    }

    intptr_t size_;
    std::vector<oor_fill_record_type> stack_;
};

}// namespace detail
}// namespace ndhist

#endif // !NDHIST_DETAIL_OOR_FILL_RECORD_STACK_HPP_INCLUDED
