/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <ndhist@martin-wolf.org>
 *
 * @brief This file provides a random-access STL iterator for iterating of the
 *        indices of a single histogram axis. This iterator can be used, e.g.
 *        for the boost::python::slice::get_indices function to get a loop over
 *        the axis indices given a certain slice.
 *
 * This file is distributed under the BSD 2-Clause Open Source License
 * (See LICENSE file).
 *
 */
#ifndef NDHIST_DETAIL_AXIS_INDEX_ITER_HPP_INCLUDED
#define NDHIST_DETAIL_AXIS_INDEX_ITER_HPP_INCLUDED 1

#include <boost/iterator/iterator_facade.hpp>

#include <ndhist/error.hpp>
#include <ndhist/detail/axis.hpp>

namespace ndhist {
namespace detail {

class axis_index_iter
  : public boost::iterator_facade<
        axis_index_iter
      , intptr_t // ValueType
      , std::random_access_iterator_tag
      , intptr_t // RefValueType
      //, DifferenceType
    >
{
  public:
    typedef boost::iterator_facade< axis_index_iter, intptr_t, std::random_access_iterator_tag, intptr_t >
            base_t;
    typedef base_t::difference_type
            difference_type;

    // Default constructor.
    axis_index_iter()
      : axis_size_(0)
      , start_index_(0)
      , end_index_(0)
      , iter_index_(0)
    {}

    // Constructor for an iterator range of [first_index, last_index]. The
    // first_index and last_index indices can be set axis::UNDERFLOW_INDEX or
    // axis::OVERFLOW_INDEX.
    explicit axis_index_iter(
        uintptr_t axis_size
      , intptr_t first_index
      , intptr_t last_index
    )
      : axis_size_(axis_size)
    {
        // Set the start index.
        if(first_index == axis::UNDERFLOW_INDEX)
        {
            start_index_ = 0;
        }
        else if(first_index == axis::OVERFLOW_INDEX)
        {
            start_index_ = axis_size_+1;
        }
        else if(first_index >= 0 && first_index < intptr_t(axis_size_))
        {
            start_index_ = first_index+1;
        }
        else // first_index >= 0 && first_index >= axis_size_
        {
            std::stringstream ss;
            ss << "The first index must be set either to the underflow or "
               << "overflow index or must be a positive value within the range "
               << "[0, "<< axis_size_-1 <<"]!";
            throw IndexError(ss.str());
        }

        // Set the end index.
        if(last_index == axis::UNDERFLOW_INDEX)
        {
            end_index_ = 1;
        }
        else if(last_index == axis::OVERFLOW_INDEX)
        {
            end_index_ = axis_size_+2;
        }
        else if(last_index >= 0 && last_index < intptr_t(axis_size_))
        {
            end_index_ = last_index+2;
        }
        else // last_index >= 0 && last_index >= axis_size_
        {
            std::stringstream ss;
            ss << "The last index must be set either to the underflow or "
               << "overflow index or must be a positive value within the range "
               << "[0, "<< axis_size_-1 <<"]!";
            throw IndexError(ss.str());
        }

        iter_index_ = start_index_;
        //std::cout << "iter_index_ = "<<iter_index_ << ", start_index_ = "<<start_index_<<", end_index_ = "<<end_index_<<std::endl;
    }

    // Creates an interator that points to the first index.
    axis_index_iter
    begin() const
    {
        axis_index_iter it(*this);
        it.iter_index_ = it.start_index_;
        return it;
    }

    // Creates an iterator that points to the index after the last index.
    axis_index_iter
    end() const
    {
        axis_index_iter it(*this);
        it.iter_index_ = it.end_index_;
        return it;
    }

    bool
    is_end() const
    {
        return (iter_index_ >= end_index_);
    }

    void
    reset()
    {
        //std::cout << "axis_index_iter: reset"<<std::endl<<std::flush;
        iter_index_ = start_index_;
    }

    void
    increment()
    {
        //std::cout << "axis_index_iter: increment"<<std::endl<<std::flush;
        if(is_end())
        {
            reset();
        }
        else
        {
            ++iter_index_;
        }
    }

    void
    decrement()
    {
        //std::cout << "axis_index_iter: decrement"<<std::endl<<std::flush;
        if(iter_index_ == start_index_)
        {
            iter_index_ = end_index_;
        }
        else
        {
            --iter_index_;
        }
    }

    bool
    equal(axis_index_iter const & other) const
    {
//         std::cout << "axis_index_iter: equal: "
//                   << "this.iter_index_ = "<< iter_index_ << ", "
//                   << "other.iter_index_ = "<< other.iter_index_
//                   << std::endl<<std::flush;
        if(axis_size_ != other.axis_size_) {
            return false;
        }
        if(is_end() && other.is_end()) {
            return true;
        }

        return (iter_index_ == other.iter_index_);
    }

    intptr_t
    dereference() const
    {
        //std::cout << "axis_index_iter: dereference"<<std::endl<<std::flush;
        if(is_end())
        {
            std::stringstream ss;
            ss << "Dereferencing out-of-bounds axis index iterator!";
            throw IndexError(ss.str());
        }
        if(iter_index_ == 0)
        {
            return axis::UNDERFLOW_INDEX;
        }
        else if(iter_index_ == intptr_t(axis_size_+1))
        {
            return axis::OVERFLOW_INDEX;
        }

        return iter_index_-1;
    }

    void
    advance(difference_type const n)
    {
        //std::cout << "axis_index_iter: advance("<<n<<")"<<std::endl<<std::flush;
        iter_index_ += n;
        if(iter_index_ > end_index_) {
            iter_index_ = end_index_;
        }
    }

    difference_type
    distance_to(axis_index_iter const & z) const
    {
        //std::cout << "axis_index_iter: distance_to"<<std::endl<<std::flush;
        difference_type d = z.iter_index_ - iter_index_;
        //std::cout << "axis_index_iter: distance_to = "<<d<<std::endl;
        return d;
    }

  protected:
    uintptr_t axis_size_;
    // The start, end and iter indices are defined in a linear space where 0
    // means the underflow bin index and axis_size_+1 the overflow bin index.
    // The indices in the range [1, axis_size_] are the indices of the axis
    // bins.
    intptr_t start_index_;
    intptr_t end_index_;
    intptr_t iter_index_;

  private:
    friend class boost::iterator_core_access;
};

}// namespace detail
}// namespace ndhist

#endif // NDHIST_DETAIL_FULL_AXIS_INDEX_ITER_HPP_INCLUDED
