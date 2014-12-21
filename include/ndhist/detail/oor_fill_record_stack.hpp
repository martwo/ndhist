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

namespace ndhist {
namespace detail {

template <typename BCValueType>
struct oor_fill_record
{
    std::vector<intptr_t> relative_indices;
    BCValueType weight;
};

class OORFillRecordStackBase
{
  public:
    OORFillRecordStackBase(intptr_t nd, intptr_t capacity)
      : nd_(nd)
      , capacity_(capacity)
    {}

    intptr_t nd_;
    intptr_t capacity_;
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
        rec.relative_indices.resize(nd);
        stack_.resize(base_t::capacity_, rec);
        std::cout << "OORFillRecordStack size = "<<stack_.size() <<std::endl;
    }

    /** Returns true, when the capacity is reached after adding the record, and
     *  false otherwise.
     */
    bool
    push_back(std::vector<intptr_t> const & relative_indices, BCValueType weight)
    {
        std::cout << "OORFillRecordStack::push_back at "<< size_ << std::endl<<std::flush;
        stack_[size_].relative_indices = relative_indices;
        stack_[size_].weight           = weight;
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
