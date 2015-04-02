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
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <sstream>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/python/list.hpp>
#include <boost/python/refcount.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/iterators/flat_iterator.hpp>
#include <boost/numpy/iterators/indexed_iterator.hpp>
#include <boost/numpy/iterators/multi_flat_iterator.hpp>
#include <boost/numpy/python/make_tuple_from_container.hpp>
#include <boost/numpy/dstream.hpp>
#include <boost/numpy/utilities.hpp>

#include <ndhist/limits.hpp>
#include <ndhist/ndhist.hpp>
#include <ndhist/axis.hpp>
#include <ndhist/type_support.hpp>
//#include <ndhist/detail/axis_index_iter.hpp>
#include <ndhist/detail/bin_iter_value_type_traits.hpp>
#include <ndhist/detail/bin_value.hpp>
#include <ndhist/detail/bin_utils.hpp>
#include <ndhist/detail/limits.hpp>
#include <ndhist/detail/multi_axis_iter.hpp>
#include <ndhist/detail/py_arg_inspector.hpp>
#include <ndhist/detail/py_seq_inspector.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace ndhist {

namespace detail {

template <typename WeightValueType>
static
void
flush_value_cache(
    ndhist                      & self
  , ValueCache<WeightValueType> & value_cache
  , std::vector<intptr_t> const & f_n_extra_bins_vec
  , uintptr_t const               bc_data_offset
)
{
    intptr_t const nd = self.get_nd();

    // Fill in the cached values.
    char * bin_data_addr;
    intptr_t idx = value_cache.get_size();
    while(idx--)
    {
        typename ValueCache<WeightValueType>::cache_entry_type const & entry = value_cache.get_entry(idx);

        std::vector<intptr_t> const & arr_strides = self.bc_.get_data_strides_vector();
        bin_data_addr = self.bc_.get_data() + bc_data_offset;

        // Translate the relative indices into an absolute
        // data address for the extended bin content array.
        for(intptr_t axis=0; axis<nd; ++axis)
        {
            bin_data_addr += (f_n_extra_bins_vec[axis] + entry.relative_indices_[axis]) * arr_strides[axis];
        }

        bin_utils<WeightValueType>::increment_bin(bin_data_addr, entry.weight_);
    }

    // Finally, clear the stack.
    value_cache.clear();
}

template <typename BCValueType>
struct iadd_fct_traits
{
    static
    void apply(ndhist & self, ndhist const & other)
    {
        if(! self.is_compatible(other))
        {
            std::stringstream ss;
            ss << "The += operator is only defined for two compatible ndhist "
               << "objects!";
            throw AssertionError(ss.str());
        }

        // Add the bin contents of the two ndhist objects.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_iter_value_type_traits<BCValueType>
                  , bin_iter_value_type_traits<BCValueType>
                >
                multi_iter_t;

        bn::ndarray self_bc_arr = self.bc_.construct_ndarray(self.bc_.get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        bn::ndarray other_bc_arr = other.bc_.construct_ndarray(other.bc_.get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        multi_iter_t bc_it(
            self_bc_arr
          , other_bc_arr
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value);
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value  = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 other_bin_value = multi_value.value_1;
            *self_bin_value.noe_  += *other_bin_value.noe_;
            *self_bin_value.sow_  += *other_bin_value.sow_;
            *self_bin_value.sows_ += *other_bin_value.sows_;
            ++bc_it;
        }
    }
};

template <typename BCValueType>
struct idiv_fct_traits
{
    static
    void apply(ndhist & self, bn::ndarray const & value_arr)
    {
        // Divide the bin contents of the ndhist object with the
        // scalar value.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_iter_value_type_traits<BCValueType>
                  , bn::iterators::single_value<BCValueType>
                >
                multi_iter_t;

        bn::ndarray self_bc_arr = self.bc_.construct_ndarray(self.bc_.get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        multi_iter_t bc_it(
            self_bc_arr
          , const_cast<bn::ndarray &>(value_arr)
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value
        );
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 value          = multi_value.value_1;
            *self_bin_value.sow_  /= value;
            *self_bin_value.sows_ /= value * value;
            ++bc_it;
        }
    }
};

template <typename BCValueType>
struct imul_fct_traits
{
    static
    void apply(ndhist & self, bn::ndarray const & value_arr)
    {
        // Multiply the bin contents of the ndhist object with the
        // scalar value.
        typedef bn::iterators::multi_flat_iterator<2>::impl<
                    bin_iter_value_type_traits<BCValueType>
                  , bn::iterators::single_value<BCValueType>
                >
                multi_iter_t;

        bn::ndarray self_bc_arr = self.bc_.construct_ndarray(self.bc_.get_dtype(), 0, /*owner=*/NULL, /*set_owndata_flag=*/false);
        multi_iter_t bc_it(
            self_bc_arr
          , const_cast<bn::ndarray &>(value_arr)
          , boost::numpy::detail::iter_operand::flags::READWRITE::value
          , boost::numpy::detail::iter_operand::flags::READONLY::value
        );
        while(! bc_it.is_end())
        {
            typename multi_iter_t::multi_references_type multi_value = *bc_it;
            typename multi_iter_t::value_ref_type_0 self_bin_value = multi_value.value_0;
            typename multi_iter_t::value_ref_type_1 value          = multi_value.value_1;
            *self_bin_value.sow_  *= value;
            *self_bin_value.sows_ *= value * value;
            ++bc_it;
        }
    }
};

template <typename WeightValueType>
struct project_fct_traits
{
    static
    ndhist
    apply(ndhist const & self, std::set<intptr_t> const & axes)
    {
        // Create a ndhist with the dimensions specified by axes.
        uintptr_t const self_nd = self.get_nd();
        uintptr_t const proj_nd = axes.size();

        bp::list axis_list;
        std::set<intptr_t>::const_iterator axes_it = axes.begin();
        std::set<intptr_t>::const_iterator const axes_end = axes.end();
        for(; axes_it != axes_end; ++axes_it)
        {
            axis_list.append(self.axes_[*axes_it]);
            std::cout << "project: got axis "<<*axes_it<<std::endl;
        }
        bp::tuple axes_tuple(axis_list);
        ndhist proj(axes_tuple, self.bc_weight_dt_, self.bc_class_);

        typedef multi_axis_iter< bin_iter_value_type_traits<WeightValueType> >
                multi_axis_iter_t;

        multi_axis_iter_t proj_iter(proj.bc_.construct_ndarray(proj.bc_.get_dtype(), /*field_idx=*/0, /*data_owner=*/NULL, /*set_owndata_flag=*/false));
        multi_axis_iter_t self_iter(self.bc_.construct_ndarray(self.bc_.get_dtype(), /*field_idx=*/0, /*data_owner=*/NULL, /*set_owndata_flag=*/false));

        // Iterate over *all* the bins (including underflow & overflow bins) of
        // the projection bin content array.
        std::vector<intptr_t> self_fixed_axes_indices(self_nd, axis::FLAGS_FLOATING_INDEX);
        std::vector<intptr_t> self_iter_axes_range_min(self_nd, 0);
        std::vector<intptr_t> self_iter_axes_range_max(self_nd);
        for(uintptr_t i=0; i<self_nd; ++i)
        {
            self_iter_axes_range_max[i] = self.bc_.get_shape_vector()[i];
        }

        std::vector<intptr_t> proj_fixed_axes_indices(proj_nd, axis::FLAGS_FLOATING_INDEX);
        std::vector<intptr_t> proj_iter_axes_range_min(proj_nd, 0);
        std::vector<intptr_t> proj_iter_axes_range_max(proj_nd);
        for(uintptr_t i=0; i<proj_nd; ++i)
        {
            proj_iter_axes_range_max[i] = proj.bc_.get_shape_vector()[i];
        }

        proj_iter.init_iteration(proj_fixed_axes_indices, proj_iter_axes_range_min, proj_iter_axes_range_max);
        while(! proj_iter.is_end())
        {
            // Get the proj bin.
            typename multi_axis_iter_t::value_ref_type proj_bin = proj_iter.dereference();

            // Iterate over all the axes of self which are not fixed through the
            // current projection indices.
            axes_it = axes.begin();
            for(uintptr_t i=0; axes_it != axes_end; ++axes_it, ++i)
            {
                self_fixed_axes_indices[*axes_it] = proj_iter.get_indices()[i];
            }

            self_iter.init_iteration(self_fixed_axes_indices, self_iter_axes_range_min, self_iter_axes_range_max);
            while(! self_iter.is_end())
            {
                // Get the self bin.
                typename multi_axis_iter_t::value_ref_type self_bin = self_iter.dereference();

                // Add the self bin to the proj bin.
                *proj_bin.noe_  += *self_bin.noe_;
                *proj_bin.sow_  += *self_bin.sow_;
                *proj_bin.sows_ += *self_bin.sows_;

                self_iter.increment();
            }

            proj_iter.increment();
        }

        return proj;
    }
};

template <typename WeightValueType>
struct merge_axis_bins_fct_traits
{
    static
    void
    apply(ndhist & self, intptr_t axis, intptr_t nbins_to_merge)
    {
        if(   nbins_to_merge == 0
           || nbins_to_merge == 1
          )
        {
            // We have nothing to do.
            return;
        }

        uintptr_t const nd = self.get_nd();

        if(   (axis < 0 && intptr_t(axis + nd) < 0)
           || (axis >= 0 && axis >= nd)
          )
        {
            std::stringstream ss;
            ss << "The axis value '"<<axis<<"' is invalid. It must be within "
               << "the interval ["<< -nd <<", "<< nd-1 <<"]!";
            throw ValueError(ss.str());
        }
        if(axis < 0)
        {
            axis += nd;
        }

        intptr_t const self_nbins = self.get_nbins()[axis];

        if(nbins_to_merge > self_nbins)
        {
            std::stringstream ss;
            ss << "The number of bins to merge for axis "<<axis<<" must be "
               << "within the interval [0, "<<self_nbins<<"]!";
            throw ValueError(ss.str());
        }

        typedef multi_axis_iter< bin_iter_value_type_traits<WeightValueType> >
                multi_axis_iter_t;

        multi_axis_iter_t rebinned_iter(self.bc_.construct_ndarray(self.bc_.get_dtype(), /*field_idx=*/0, /*data_owner=*/NULL, /*set_owndata_flag=*/false));
        multi_axis_iter_t self_iter(self.bc_.construct_ndarray(self.bc_.get_dtype(), /*field_idx=*/0, /*data_owner=*/NULL, /*set_owndata_flag=*/false));

        // Iterate over the new rebinned indices (excluding the underflow and
        // overflow bin).
        std::vector<intptr_t> rebinned_fixed_axes_indices(nd, axis::FLAGS_FLOATING_INDEX);
        std::vector<intptr_t> rebinned_iter_axes_range_min(nd, 0);
        std::vector<intptr_t> rebinned_iter_axes_range_max(nd);

        std::vector<intptr_t> self_fixed_axes_indices(nd, axis::FLAGS_FLOATING_INDEX);
        std::vector<intptr_t> self_iter_axes_range_min(nd, 0);
        std::vector<intptr_t> self_iter_axes_range_max(nd);

        for(uintptr_t i=0; i<nd; ++i)
        {
            intptr_t const nbins = self.bc_.get_shape_vector()[i];
            rebinned_iter_axes_range_max[i] = nbins;
            self_iter_axes_range_max[i] = nbins;
        }

        bool const is_extendable_axis = self.axes_[axis]->is_extendable();
        intptr_t const rebinned_nbins = self_nbins / nbins_to_merge;
        intptr_t const offset = self.axes_[axis]->has_underflow_bin() ? 1 : 0;
        intptr_t const rebinned_end_idx = offset + rebinned_nbins;
        intptr_t rebinned_idx = offset;
        intptr_t idx = 0;
        for(; rebinned_idx < rebinned_end_idx; ++rebinned_idx, ++idx)
        {
            // Fix the rebinned_idx of axis for the rebinned_iter.
            rebinned_fixed_axes_indices[axis] = rebinned_idx;
            rebinned_iter.init_iteration(rebinned_fixed_axes_indices, rebinned_iter_axes_range_min, rebinned_iter_axes_range_max);
            while(! rebinned_iter.is_end())
            {
                // We need to reset the bin to zero if it's not the first bin,
                // because the first bin is one of the bins to sum over.
                if(rebinned_idx != offset)
                {
                    bin_utils<WeightValueType>::zero_bin(rebinned_iter.get_data());
                }

                // Get the (zeroed) rebinned bin.
                typename multi_axis_iter_t::value_ref_type rebinned_bin = rebinned_iter.dereference();

                // This bin contains the sum of the merged bins specified by
                // the current other bin indicies.
                for(uintptr_t i=0; i<nd; ++i)
                {
                    if(i != axis)
                    {
                        // The index for axis keeps always floating.
                        self_fixed_axes_indices[i] = rebinned_iter.get_indices()[i];
                    }
                }
                // Specify the index iteration range for the axis. Keep in mind
                // that the first rebinned bin is not reset to zero, because
                // it one of the bins to sum over.
                self_iter_axes_range_min[axis] = offset + idx * nbins_to_merge + (rebinned_idx == offset);
                self_iter_axes_range_max[axis] = offset + idx * nbins_to_merge + nbins_to_merge;

                self_iter.init_iteration(self_fixed_axes_indices, self_iter_axes_range_min, self_iter_axes_range_max);
                while(! self_iter.is_end())
                {
                    // Get the self bin.
                    typename multi_axis_iter_t::value_ref_type self_bin = self_iter.dereference();

                    // Add the self bin to the rebinned bin.
                    *rebinned_bin.noe_  += *self_bin.noe_;
                    *rebinned_bin.sow_  += *self_bin.sow_;
                    *rebinned_bin.sows_ += *self_bin.sows_;

                    if(is_extendable_axis) {
                        bin_utils<WeightValueType>::zero_bin(self_iter.get_data());
                    }

                    self_iter.increment();
                }

                rebinned_iter.increment();
            }
        }

        // Move the overflow bin and merge it with the remaining bins.
        // This can only be done, when the axis is not extendable, because
        // extendable axes don't have an overflow bin.
        // If the axis does not provide an overflow bin, an overflow bin will
        // be created, where the edge value is taken from the right most bin
        // used to merge into the overflow bin.

        // Calculate the number of bins that will fall into the overflow bin (in
        // case the axis contains an overflow bin).
        bool const self_axis_has_overflow_bin = self.axes_[axis]->has_overflow_bin();
        bool rebinned_axis_has_overflow_bin = false;
        intptr_t const nbins_into_overflow = self_nbins % nbins_to_merge;
        if(   !is_extendable_axis
           && (self_axis_has_overflow_bin || nbins_into_overflow > 0)
          )
        {
            rebinned_axis_has_overflow_bin = true;
            rebinned_idx = rebinned_end_idx;

            // Define the sum over indices.
            self_iter_axes_range_min[axis] = self.axes_[axis]->get_n_bins() - self_axis_has_overflow_bin - nbins_into_overflow;
            self_iter_axes_range_max[axis] = self.axes_[axis]->get_n_bins();

            rebinned_fixed_axes_indices[axis] = rebinned_idx;
            rebinned_iter.init_iteration(rebinned_fixed_axes_indices, rebinned_iter_axes_range_min, rebinned_iter_axes_range_max);
            while(! rebinned_iter.is_end())
            {
                // Zero the current rebinned bin.
                bin_utils<WeightValueType>::zero_bin(rebinned_iter.get_data());

                // Get the zeroed rebinned bin.
                typename multi_axis_iter_t::value_ref_type rebinned_bin = rebinned_iter.dereference();

                // This bin contains the sum of the merged bins specified by
                // the current other bin indicies.
                for(uintptr_t i=0; i<nd; ++i)
                {
                    if(i != axis)
                    {
                        // The index for axis keeps always floating.
                        self_fixed_axes_indices[i] = rebinned_iter.get_indices()[i];
                    }
                }

                self_iter.init_iteration(self_fixed_axes_indices, self_iter_axes_range_min, self_iter_axes_range_max);
                while(! self_iter.is_end())
                {
                    // Get the self bin.
                    typename multi_axis_iter_t::value_ref_type self_bin = self_iter.dereference();

                    // Add the self bin to the rebinned bin.
                    *rebinned_bin.noe_  += *self_bin.noe_;
                    *rebinned_bin.sow_  += *self_bin.sow_;
                    *rebinned_bin.sows_ += *self_bin.sows_;

                    if(is_extendable_axis) {
                        bin_utils<WeightValueType>::zero_bin(self_iter.get_data());
                    }

                    self_iter.increment();
                }

                rebinned_iter.increment();
            }
        }
        if(is_extendable_axis && nbins_into_overflow > 0)
        {
            // The nbins_into_overflow bins will be discarded due to the
            // nature of the extendable axis. But we need to zero those bins,
            // so they won't have a pre-set value when the axis gets extended
            // afterwards.

            // Iterate over the nbins_into_overflow bins and zero them.
            for(uintptr_t i=0; i<nd; ++i)
            {
                 self_fixed_axes_indices[i] = axis::FLAGS_FLOATING_INDEX;
            }
            self_iter_axes_range_min[axis] = self.axes_[axis]->get_n_bins() - 1 - nbins_into_overflow;
            self_iter_axes_range_max[axis] = self.axes_[axis]->get_n_bins();
            self_iter.init_iteration(self_fixed_axes_indices, self_iter_axes_range_min, self_iter_axes_range_max);
            while(! self_iter.is_end())
            {
                bin_utils<WeightValueType>::zero_bin(self_iter.get_data());
                self_iter.increment();
            }
        }

        // Adjust the data view (i.e. shape and back capacity) for the axis.
        bool const rebinned_axis_has_underflow_bin = self.axes_[axis]->has_underflow_bin();
        intptr_t const rebinned_axis_shape = rebinned_axis_has_underflow_bin + rebinned_nbins + rebinned_axis_has_overflow_bin;
        std::vector<intptr_t> delta_shape(nd, 0);
        std::vector<intptr_t> delta_front_capacity(nd, 0);
        std::vector<intptr_t> delta_back_capacity(nd, 0);
        delta_shape[axis] = rebinned_axis_shape - self.bc_.get_shape_vector()[axis];
        delta_back_capacity[axis] = -delta_shape[axis];
        self.bc_.change_view(delta_shape, delta_front_capacity, delta_back_capacity);

        // Adjust the bin edges of the axis.
        bn::ndarray const oldedges = self.axes_[axis]->get_binedges_ndarray();
        std::vector<intptr_t> const shape(1, rebinned_axis_shape+1);
        bn::ndarray newedges = bn::empty(shape, self.axes_[axis]->get_dtype());
        typedef bn::iterators::flat_iterator< bn::iterators::single_value<WeightValueType> >
                edges_iter_t;
        edges_iter_t oldedges_iter(oldedges);
        edges_iter_t newedges_iter(newedges);
        if(rebinned_axis_has_underflow_bin)
        {
            // The first edge is the underflow bin lower edge.
            newedges_iter.set_value(*oldedges_iter);
            ++oldedges_iter;
            ++newedges_iter;
        }
        for(intptr_t i=0; i<rebinned_nbins; ++i)
        {
            // Set the lower edge of the current visible bin.
            newedges_iter.set_value(*oldedges_iter);
            oldedges_iter.advance(nbins_to_merge);
            ++newedges_iter;
        }
        if(rebinned_axis_has_overflow_bin)
        {
            // Set the lower edge of the overflow bin.
            newedges_iter.set_value(*oldedges_iter);
            oldedges_iter.advance(self_iter_axes_range_max[axis] - self_iter_axes_range_min[axis]);
            ++newedges_iter;
        }
        // Set the upper edge of the last bin.
        newedges_iter.set_value(*oldedges_iter);

        // Create a new axis object for the changed axis using the new edge
        // array.
        Axis const & oldaxis = *self.axes_[axis];
        self.axes_[axis] = oldaxis.create(
            newedges
          , oldaxis.get_label()
          , oldaxis.get_name()
          , rebinned_axis_has_underflow_bin
          , rebinned_axis_has_overflow_bin
          , oldaxis.is_extendable()
          , oldaxis.get_extension_max_fcap()
          , oldaxis.get_extension_max_bcap()
        );
    }
};

template <typename WeightValueType>
struct clear_fct_traits
{
    static
    void
    apply(ndhist & self)
    {
        if(! self.is_view())
        {
            // This ndhist object is not a view and the bin content array holds
            // POD values, so we can just memset the entire bin content array.
            self.bc_.clear();
            return;
        }

        // This ndhist object is a view on only a part of the bin content
        // array, so we need to iterate over the view's bins and set them to
        // zero.
        typedef multi_axis_iter< bin_iter_value_type_traits<WeightValueType> >
                multi_axis_iter_t;

        multi_axis_iter_t self_iter(self.bc_.construct_ndarray(self.bc_.get_dtype(), /*field_idx=*/0, /*data_owner=*/NULL, /*set_owndata_flag=*/false));
        self_iter.init_full_iteration();
        while(! self_iter.is_end())
        {
            bin_utils<WeightValueType>::zero_bin(self_iter.get_data());

            self_iter.increment();
        }
    }
};

template <>
struct clear_fct_traits<bp::object>
{
    static
    void
    apply(ndhist & self)
    {
        // This ndhist object holds Python objects as bin content values, in
        // order to set the values to zero and keeping the reference count of
        // the Python objects valid, we need to iterate over each bin and set
        // it to zero individually.
        typedef multi_axis_iter< bin_iter_value_type_traits<bp::object> >
                multi_axis_iter_t;

        multi_axis_iter_t self_iter(self.bc_.construct_ndarray(self.bc_.get_dtype(), /*field_idx=*/0, /*data_owner=*/NULL, /*set_owndata_flag=*/false));
        self_iter.init_full_iteration();
        while(! self_iter.is_end())
        {
            bin_utils<bp::object>::zero_bin(self_iter.get_data());

            self_iter.increment();
        }
    }
};

/**
 * @brief Creates a ND-sized vector of ndarray objects which are views into the
 *     complete (i.e. including under- and overflow bins) bin content array.
 *     So, for example, for the first array, the index of the first dimension
 *     is fixed (either 0 for underflow bins or n_bins for overflow bins), and
 *     the indices of the other dimensions of the first array are not fixed. So
 *     the shape of the first array is (1, n_bins_y+2, n_bins_z+2, ...).
 * @note The returned ndarray object don't have the owndata flag set and have
 *     also no base object set. So they should be regarded as internal objects.
 *     In case they are handed out to the user, their base object needs to be
 *     set.
 */
template <typename BCValueType>
static
std::vector<bn::ndarray>
get_field_axes_oor_ndarrays(
    ndhist const & self
  , ::ndhist::axis::out_of_range_t const oortype
  , size_t const field_idx
)
{
    uintptr_t const nd = self.get_nd();
    std::vector<intptr_t> complete_bc_arr_shape          = self.bc_.get_shape_vector();
    std::vector<intptr_t> complete_bc_arr_front_capacity = self.bc_.get_front_capacity_vector();
    std::vector<intptr_t> complete_bc_arr_back_capacity  = self.bc_.get_back_capacity_vector();
    // Add the under- and overflow bins of the extendable axes to the shape, and
    // remove them from the front- and back capacities, in order to calculate
    // the data offset and strides correctly.
    for(uintptr_t i=0; i<nd; ++i)
    {
        if(self.axes_[i]->is_extendable())
        {
            complete_bc_arr_shape[i] += 2;
            complete_bc_arr_front_capacity[i] -= 1;
            complete_bc_arr_back_capacity[i] -= 1;
        }
        else
        {
            // Check if any axis does not provide under or overflow bins at all.
            if(   (oortype == ::ndhist::axis::OOR_UNDERFLOW && !self.axes_[i]->has_underflow_bin())
               || (oortype == ::ndhist::axis::OOR_OVERFLOW  && !self.axes_[i]->has_overflow_bin())
              )
            {
                std::stringstream ss;
                ss << "Axis "<<i<<" does not provide an "
                << (oortype == ::ndhist::axis::OOR_UNDERFLOW ? "underflow" : "overflow") << " bin! "
                << "In case the ndhist object is a data view into an other "
                << "ndhist object, this is expected and thus not an error!";
                throw AssertionError(ss.str());
            }
        }
    }

    intptr_t const sub_item_byte_offset = (field_idx == 0 ? 0 : self.bc_.get_dtype().get_fields_byte_offsets()[field_idx]);

    // Allocate vectors for the shape, front and back capacities that are used
    // to construct the views of the returned individual ndarrays.
    std::vector<intptr_t> shape(nd);
    std::vector<intptr_t> front_capacity(nd);
    std::vector<intptr_t> back_capacity(nd);

    bn::dtype const dt = (field_idx == 0 ? self.bc_noe_dt_ : self.bc_weight_dt_);

    std::vector<bn::ndarray> array_vec;
    array_vec.reserve(nd);
    for(uintptr_t i=0; i<nd; ++i)
    {
        for(uintptr_t j=0; j<nd; ++j)
        {
            if(j == i)
            {
                // When we get to the i'th ndarray the axis consists of only one
                // index.
                shape[j] = 1;
                if(oortype == ::ndhist::axis::OOR_UNDERFLOW)
                {
                    front_capacity[j] = complete_bc_arr_front_capacity[j];
                    back_capacity[j] = complete_bc_arr_back_capacity[j] + complete_bc_arr_shape[j] - 1;
                }
                else // oortype == ::ndhist::axis::OOR_OVERFLOW
                {
                    front_capacity[j] = complete_bc_arr_front_capacity[j] + complete_bc_arr_shape[j] - 1;
                    back_capacity[j] = complete_bc_arr_back_capacity[j];
                }
            }
            else
            {
                shape[j] = complete_bc_arr_shape[j];
                front_capacity[j] = complete_bc_arr_front_capacity[j];
                back_capacity[j] = complete_bc_arr_back_capacity[j];
            }
        }

        // Construct the ndarray, that is a view into the bin content array.
        bn::ndarray arr = ndarray_storage::construct_ndarray(self.bc_, dt, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
        array_vec.push_back(arr);
    }

    return array_vec;
}

/*
struct generic_nd_traits
{
    template <typename BCValueType>
    struct fill_fct_traits
    {
        static
        void
        apply(ndhist & self, bp::object const & ndvalues_obj, bp::object const & weight_obj)
        {
            // The ndvalues_obj object is supposed to be a structured ndarray.
            // But in case of 1-dimensional histogram we accept also a
            // simple array, which can also be a normal Python list.

            size_t const nd = self.get_nd();
            bp::object ndvalues_arr_obj;
            try
            {
                ndvalues_arr_obj = bn::from_object(ndvalues_obj, self.get_ndvalues_dtype(), 0, 0, bn::ndarray::ALIGNED);
            }
            catch (const bp::error_already_set &)
            {
                if(nd == 1)
                {
                    ndvalues_arr_obj = bn::from_object(ndvalues_obj, self.get_axes()[0]->get_dtype(), 0, 0, bn::ndarray::ALIGNED);
                }
                else
                {
                    std::stringstream ss;
                    ss << "The ndvalues parameter must either be a tuple of "
                       << nd << " one-dimensional arrays or one structured "
                       << "ndarray!";
                    throw TypeError(ss.str());
                }
            }
            bn::ndarray ndvalue_arr = *static_cast<bn::ndarray*>(&ndvalues_arr_obj);
            bn::ndarray weight_arr = bn::from_object(weight_obj, bn::dtype::get_builtin<BCValueType>(), 0, 0, bn::ndarray::ALIGNED);

            // Construct an iterator for the input arrays. We use the loop service
            // of BoostNumpy that determines the number of loop
            // dimensions automatically and provides generalized universal
            // functions.
            typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
                    core_shape_t0;
            typedef bn::dstream::array_definition< core_shape_t0, void>
                    in_arr_def0;
            typedef bn::dstream::mapping::detail::core_shape<0>::shape<>
                    core_shape_t1;
            typedef bn::dstream::array_definition< core_shape_t1, BCValueType>
                    in_arr_def1;
            typedef bn::dstream::detail::loop_service_arity<2>::loop_service<in_arr_def0, in_arr_def1>
                    loop_service_t;

            bn::dstream::detail::input_array_service<in_arr_def0> in_arr_service0(ndvalue_arr);
            bn::dstream::detail::input_array_service<in_arr_def1> in_arr_service1(weight_arr);
            loop_service_t loop_service(in_arr_service0, in_arr_service1);

            bn::detail::iter_operand_flags_t in_arr_iter_op_flags0 = bn::detail::iter_operand::flags::READONLY::value;
            bn::detail::iter_operand_flags_t in_arr_iter_op_flags1 = bn::detail::iter_operand::flags::READONLY::value;

            bn::detail::iter_operand in_arr_iter_op0( in_arr_service0.get_arr(), in_arr_iter_op_flags0, in_arr_service0.get_arr_bcr_data() );
            bn::detail::iter_operand in_arr_iter_op1( in_arr_service1.get_arr(), in_arr_iter_op_flags1, in_arr_service1.get_arr_bcr_data() );

            bn::detail::iter_flags_t iter_flags =
                bn::detail::iter::flags::REFS_OK::value // This is needed for the
                                                        // weight, which can be bp::object.
              | bn::detail::iter::flags::EXTERNAL_LOOP::value;
            bn::order_t order = bn::KEEPORDER;
            bn::casting_t casting = bn::NO_CASTING;
            intptr_t buffersize = 0;

            bn::detail::iter iter(
                  iter_flags
                , order
                , casting
                , loop_service.get_loop_nd()
                , loop_service.get_loop_shape_data()
                , buffersize
                , in_arr_iter_op0
                , in_arr_iter_op1
            );
            iter.init_full_iteration();

            // Create an indexed iterator for the bin content array.
            bp::object self_obj(bp::ptr(&self));
            bn::ndarray bc_arr = self.bc_->ConstructNDArray(self.bc_weight_dt_, 1, &self_obj);
            bn::iterators::indexed_iterator< bn::iterators::single_value<BCValueType> > bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);

            // Get the byte offsets of the fields and check if the number of fields
            // match the dimensionality of the histogram.
            std::vector<intptr_t> ndvalue_byte_offsets = ndvalue_arr.get_dtype().get_fields_byte_offsets();

            if(ndvalue_byte_offsets.size() == 0)
            {
                if(nd == 1)
                {
                    ndvalue_byte_offsets.push_back(0);
                }
                else
                {
                    std::stringstream ss;
                    ss << "The dimensionality of the histogram is " << nd
                       << ", i.e. greater than 1, so the value ndarray must be a "
                       << "structured ndarray!";
                    throw ValueError(ss.str());
                }
            }
            else if(ndvalue_byte_offsets.size() != nd)
            {
                std::stringstream ss;
                ss << "The value ndarray must contain " << nd << " fields, one for "
                   << "each dimension! Right now it has "
                   << ndvalue_byte_offsets.size() << " fields!";
                throw ValueError(ss.str());
            }

            // Do the iteration.
            typedef typename bin_utils<BCValueType>::ref_type
                    bc_ref_type;
            std::vector<intptr_t> indices(nd);
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Get the weight scalar from the iterator.
                    bc_ref_type weight = bin_utils<BCValueType>::get_weight_type_value_from_iter(iter, 1);

                    // Fill the scalar into the bin content array.
                    // Get the coordinate of the current ndvalue.
                    bool is_overflown = false;
                    for(size_t i=0; i<nd; ++i)
                    {
                        //std::cout << "Get bin idx of axis " << i << " of " << nd << std::endl;
                        boost::shared_ptr<detail::Axis> & axis = self.get_axes()[i];
                        char * ndvalue_ptr = iter.get_data(0) + ndvalue_byte_offsets[i];
                        axis::out_of_range_t oor;
                        intptr_t axis_idx = axis->get_bin_index_fct(axis->data_, ndvalue_ptr, &oor);
                        if(oor == axis::OOR_NONE)
                        {
                            indices[i] = axis_idx;
                        }
                        else
                        {
                            // The current value is out of the axis bounds.
                            // Just ignore it for now.
                            // TODO: Introduce an under- and overflow bin for each
                            //       each axis. Or resize the axis.
                            is_overflown = true;
                            break;
                        }
                    }

                    // Increase the bin content if the bin exists.
                    if(!is_overflown)
                    {
                        bc_iter.jump_to(indices);
                        bc_ref_type bc_value = *bc_iter;
                        bc_value += weight;
                    }

                    // Jump to the next fill iteration.
                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    }; // struct fill_traits
}; // struct generic_nd_traits
*/

template <int nd>
struct specific_nd_traits;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND, <ndhist/ndhist.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail

ndhist::
ndhist(
    bp::tuple const & axes
  , bp::object const & dt
  , bp::object const & bc_class
)
  : nd_(bp::len(axes))
  , ndvalues_dt_(bn::dtype::new_builtin<void>())
  , bc_noe_dt_(bn::dtype::get_builtin<uintptr_t>())
  , bc_weight_dt_(bn::dtype(dt))
  , bc_class_(bc_class)
{
    std::vector<intptr_t> shape(nd_);
    axes_extension_max_fcap_vec_.resize(nd_);
    axes_extension_max_bcap_vec_.resize(nd_);

    // Get the axes of the histogram.
    for(size_t i=0; i<nd_; ++i)
    {
        boost::shared_ptr<Axis> axis = bp::extract< boost::shared_ptr<Axis> >(axes[i]);

        // Set an axis name if it is not specified.
        std::string & axis_name = axis->get_name();
        if(axis_name == "")
        {
            std::stringstream ss_axis_name;
            ss_axis_name << "a" << i;
            axis_name = ss_axis_name.str();
        }

        // Add the axis field to the ndvalues dtype object.
        ndvalues_dt_.add_field(axis_name, axis->get_dtype());

        // Add the bin content shape information for this axis. The number of
        // of bins of an axis include possible under- and overflow bins.
        shape[i] = axis->get_n_bins();

        // Set the extra front and back capacity for this axis if the axis is
        // extendable. The +1 is for the under- or overflow bin, which will be
        // always zero but important to have when making a view to the under-
        // and overflow bin arrays.
        intptr_t & axis_max_fcap = * axis->get_extension_max_fcap_ptr();
        intptr_t & axis_max_bcap = * axis->get_extension_max_bcap_ptr();
        if(axis->is_extendable())
        {
            axis_max_fcap += 1;
            axis_max_bcap += 1;
        }
        else
        {
            axis_max_fcap = 0;
            axis_max_bcap = 0;
        }
        axes_extension_max_fcap_vec_[i] = axis_max_fcap;
        axes_extension_max_bcap_vec_[i] = axis_max_bcap;

        // Add the axis to the axes vector.
        axes_.push_back(axis);
    }

    // TODO: Make this as an option in the constructor.
    intptr_t value_cache_size = 65536;

    // Create a ndarray_storage for the bin content array. Each bin content
    // element consists of three sub-elements:
    // number_of_entries (noe), sum_of_weights (sow), and sum_of_weights_squared
    // (sows).
    bn::dtype bc_dt = bn::dtype::new_builtin<void>();
    bc_dt.add_field("noe",  bc_noe_dt_);
    bc_dt.add_field("sow",  bc_weight_dt_);
    bc_dt.add_field("sows", bc_weight_dt_);
    bc_ = detail::ndarray_storage(bc_dt, shape, axes_extension_max_fcap_vec_, axes_extension_max_bcap_vec_);

    // Setup the function pointers and the value cache.
    setup_function_pointers();
    setup_value_cache(value_cache_size);

    // Initialize the bin content array with objects using their default
    // constructor when the bin content array is an object array.
    if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
    {
        bn::ndarray bc_arr = construct_complete_bin_content_ndarray(bc_.get_dtype());
        bn::iterators::flat_iterator< detail::bin_iter_value_type_traits<bp::object> > bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);
        while(! bc_iter.is_end())
        {
            detail::bin_iter_value_type_traits<bp::object>::value_ref_type bin = *bc_iter;

            bp::object sow_obj  = bc_class_();
            bp::object sows_obj = bc_class_();
            *bin.sow_obj_ptr_  = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sow_obj.ptr()));
            *bin.sows_obj_ptr_ = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(sows_obj.ptr()));

            ++bc_iter;
        }
    }
}

ndhist::
ndhist(
    ndhist const & base
  , std::vector< boost::shared_ptr<Axis> > const & axes
  , intptr_t const bytearray_data_offset
  , std::vector<intptr_t> const & data_shape
  , std::vector<intptr_t> const & data_strides
)
  : nd_(axes.size())
  , ndvalues_dt_(bn::dtype::new_builtin<void>())
  , bc_noe_dt_(bn::dtype::get_builtin<uintptr_t>())
  , bc_weight_dt_(base.get_weight_dtype())
  , bc_class_(base.get_weight_class())
  , base_(base.shared_from_this())
{
    if(data_shape.size() != data_strides.size())
    {
        std::stringstream ss;
        ss << "The lengths of the data_shape and data_strides vectors must "
           << "match! Currently, they are "<< data_shape.size() <<" and "
           << data_strides.size() << ", respectively!";
        throw AssertionError(ss.str());
    }

    // Setup the axes related properties of ndhist.
    for(size_t i=0; i<nd_; ++i)
    {
        boost::shared_ptr<Axis> const & axis = axes[i];

        // Check that the is_extendable flag is set
        // to false for the axis.
        if(! (   axis->is_extendable() == false
              && axis->get_extension_max_fcap() == 0
              && axis->get_extension_max_bcap() == 0
             ))
        {
            std::stringstream ss;
            ss << "The axis "<<i<<" must have the is_extendable flag set to "
               << "false, and the maximal front and back capacities set to 0!";
            throw AssertionError(ss.str());
        }

        ndvalues_dt_.add_field(axis->get_name(), axis->get_dtype());

        axes_.push_back(axis);
    }

    // For a data view no front and back capacity is allowed, because the bins
    // of the additional front and back capacities could overlap with existing
    // bins.
    axes_extension_max_fcap_vec_.resize(nd_, 0);
    axes_extension_max_bcap_vec_.resize(nd_, 0);

    // Create a ndarray_storage object that shares its internal data with the
    // bin content array ndarray_storage object of the owner ndhist object.
    bc_ = base.bc_.create_data_view(bytearray_data_offset, data_shape, data_strides);

    setup_function_pointers();
    setup_value_cache(base.value_cache_->get_capacity());
}

ndhist::
~ndhist()
{
    std::cout << "Destructing ndhist"<<std::endl;
    // If the bin content array is an object array, we need to decref the
    // objects we have created (and incref'ed) at construction time.
    if(!is_view() && bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
    {
        bn::ndarray bc_arr = construct_complete_bin_content_ndarray(bc_.get_dtype());
        bn::iterators::flat_iterator< detail::bin_iter_value_type_traits<bp::object> > bc_iter(bc_arr, bn::detail::iter_operand::flags::READWRITE::value);
        while(! bc_iter.is_end())
        {
            detail::bin_iter_value_type_traits<bp::object>::value_ref_type bin = *bc_iter;

            bp::decref<PyObject>(reinterpret_cast<PyObject*>(*bin.sow_obj_ptr_));
            bp::decref<PyObject>(reinterpret_cast<PyObject*>(*bin.sows_obj_ptr_));
            ++bc_iter;
        }
    }
}

boost::shared_ptr<ndhist>
ndhist::
deepcopy() const
{
    // Use the default copy constructor, that makes a shallow copy.
    boost::shared_ptr<ndhist> thecopy = boost::shared_ptr<ndhist>(new ndhist(*this));

    // Copy the bytearray, if this ndhist object is not a view.
    std::cout << "ndhist::copy: deepcopying bytearray ..."<<std::flush;
    thecopy->bc_.bytearray_ = bc_.bytearray_->deepcopy();
    std::cout << "done."<<std::endl<<std::flush;

    // Reset the base object. A deep copy is not a view anymore.
    thecopy->base_ = boost::shared_ptr<ndhist>();

    // Copy the value cache.
    std::cout << "ndhist::copy: deepcopying value cache ..."<<std::flush;
    thecopy->value_cache_ = value_cache_->deepcopy();
    std::cout << "done."<<std::endl<<std::flush;

    // Copy the Axes objects.
    for(uintptr_t i=0; i<nd_; ++i)
    {
        std::cout << "ndhist::copy: deepcopying axis "<<i<<" ..."<<std::flush;
        thecopy->axes_[i] = axes_[i]->deepcopy();
        std::cout << "done."<<std::endl<<std::flush;
    }

    return thecopy;
}

void
ndhist::
setup_function_pointers()
{
    // Set the fill function based on the weight data type.
    bool bc_dtype_supported = false;
    #define BOOST_PP_ITERATION_PARAMS_1                                     \
        (4, (1, NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND, <ndhist/ndhist.hpp>, 2))
    #include BOOST_PP_ITERATE()
    else
    {
        // nd is greater than NDHIST_DETAIL_LIMIT_TUPLE_FILL_MAX_ND.
        #define NDHIST_WEIGHT_VALUE_TYPE_SUPPORT(r, data, WEIGHT_VALUE_TYPE)\
            if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<WEIGHT_VALUE_TYPE>()))\
            {                                                               \
                if(bc_dtype_supported)                                      \
                {                                                           \
                    std::stringstream ss;                                   \
                    ss << "The bin content data type is supported by more than "\
                       << "one possible C++ data type! This is an internal "\
                       << "error!";                                         \
                    throw TypeError(ss.str());                              \
                }                                                           \
                /*fill_fct_ = &detail::generic_nd_traits::fill_fct_traits<WEIGHT_VALUE_TYPE>::apply;*/\
                bc_dtype_supported = true;                                  \
            }
        BOOST_PP_SEQ_FOR_EACH(NDHIST_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)
        #undef NDHIST_WEIGHT_VALUE_TYPE_SUPPORT
    }
    if(!bc_dtype_supported)
    {
        std::stringstream ss;
        ss << "The data type of the bin content array is not supported.";
        throw TypeError(ss.str());
    }

    #define NDHIST_WEIGHT_VALUE_TYPE_SUPPORT(r, data, WEIGHT_VALUE_TYPE)    \
        if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<WEIGHT_VALUE_TYPE>()))\
        {                                                                   \
            iadd_fct_ = &detail::iadd_fct_traits<WEIGHT_VALUE_TYPE>::apply; \
            idiv_fct_ = &detail::idiv_fct_traits<WEIGHT_VALUE_TYPE>::apply; \
            imul_fct_ = &detail::imul_fct_traits<WEIGHT_VALUE_TYPE>::apply; \
            get_weight_type_field_axes_oor_ndarrays_fct_ = &detail::get_field_axes_oor_ndarrays<WEIGHT_VALUE_TYPE>;\
            project_fct_ = &detail::project_fct_traits<WEIGHT_VALUE_TYPE>::apply;\
            merge_axis_bins_fct_ = &detail::merge_axis_bins_fct_traits<WEIGHT_VALUE_TYPE>::apply;\
            clear_fct_ = &detail::clear_fct_traits<WEIGHT_VALUE_TYPE>::apply;\
        }
    BOOST_PP_SEQ_FOR_EACH(NDHIST_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)
    #undef NDHIST_WEIGHT_VALUE_TYPE_SUPPORT

    get_noe_type_field_axes_oor_ndarrays_fct_ = &detail::get_field_axes_oor_ndarrays<uintptr_t>;
}

void
ndhist::
setup_value_cache(intptr_t const value_cache_size)
{
    #define NDHIST_WEIGHT_VALUE_TYPE_SUPPORT(r, data, WEIGHT_VALUE_TYPE)    \
        if(bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<WEIGHT_VALUE_TYPE>()))\
        {                                                                   \
            value_cache_ = boost::shared_ptr< detail::ValueCache<WEIGHT_VALUE_TYPE> >(new detail::ValueCache<WEIGHT_VALUE_TYPE>(nd_, value_cache_size));\
        }
    BOOST_PP_SEQ_FOR_EACH(NDHIST_WEIGHT_VALUE_TYPE_SUPPORT, ~, NDHIST_TYPE_SUPPORT_WEIGHT_VALUE_TYPES)
    #undef NDHIST_WEIGHT_VALUE_TYPE_SUPPORT
}

void
ndhist::
calc_core_bin_content_ndarray_settings(
    std::vector<intptr_t> & shape
  , std::vector<intptr_t> & front_capacity
  , std::vector<intptr_t> & back_capacity
) const
{
    shape          = bc_.get_shape_vector();
    front_capacity = bc_.get_front_capacity_vector();
    back_capacity  = bc_.get_back_capacity_vector();
    for(uintptr_t i=0; i<nd_; ++i)
    {
        // Note, that extendable axes have only virtual under- and
        // overflow bins, that are included in the front and back
        // capacities.
        if(!axes_[i]->is_extendable())
        {
            if(axes_[i]->has_underflow_bin())
            {
                shape[i] -= 1;
                front_capacity[i] += 1;
            }
            if(axes_[i]->has_overflow_bin())
            {
                shape[i] -= 1;
                back_capacity[i] += 1;
            }
        }
    }
}

ndhist &
ndhist::operator+=(ndhist const & rhs)
{
    iadd_fct_(*this, rhs);
    return *this;
}

ndhist
ndhist::
operator+(ndhist const & rhs) const
{
    if(! this->is_compatible(rhs))
    {
        std::stringstream ss;
        ss << "The right-hand-side ndhist object for the + operator must be "
           << "compatible with this ndhist object!";
        throw AssertionError(ss.str());
    }

    ndhist newhist = this->empty_like();
    newhist += *this;
    newhist += rhs;

    return newhist;
}

ndhist
ndhist::
operator[](bp::object const & arg) const
{
    // According to the indexing documentation of numpy, basic indexing occures
    // when arg is a slice object, an integer, or a tuple of slice
    // objects or integers. Basic indexing is also initiated, when arg is a
    // list of slice objects. Note, a list of integers does NOT initiate basic
    // indexing!
    //
    // Check if we have to initiate basic indexing.
    detail::py::arg_inspector arginsp(arg);
    if(   arginsp.is_of_type(PyInt_Type, PySlice_Type, PyEllipsis_Type)
       || arginsp.is_tuple_of(PyInt_Type, PySlice_Type, PyEllipsis_Type)
       || (arginsp.is_list_of(PyInt_Type, PySlice_Type, PyEllipsis_Type) && !arginsp.is_list_of(PyInt_Type))
      )
    {
        // The given argument initializes basic indexing.
        bp::object seq;

        // First convert a single object into a tuple of length one.
        // Note, if arg is a list, we keep it as a list.
        if(arginsp.is_of_type(PyInt_Type, PySlice_Type, PyEllipsis_Type))
        {
            std::cout << "Make tuple from scalar value"<<std::endl;
            seq = bp::make_tuple<bp::object>(arg);
        }
        else
        {
            seq = arg;
        }

        size_t const nsdim = bp::len(seq);

        // We are not supporting newaxis, so the number of slice dimensions
        // (nsdim) cannot be greater than the dimensionality of the histogram.
        if(nsdim > nd_)
        {
            std::stringstream ss;
            ss << "The number of slice dimensions must not exceed the "
               << "dimensionality of the histogram ("<< nd_ << ")!";
            throw IndexError(ss.str());
        }

        // Check that there is only one possible ellipsis object in the slicing
        // sequence.
        detail::py::seq_inspector seqinsp(seq);
        if(! seqinsp.has_unique_object_of_type(PyEllipsis_Type))
        {
            std::stringstream ss;
            ss << "There must be at most one ellipse object present in the "
               << "slicing sequence!";
            throw IndexError(ss.str());
        }

        // Loop over the slicing sequence and construct the axes of the new
        // (sub) histogram.
        std::vector< boost::shared_ptr<Axis> > axes;
        std::vector<intptr_t> data_shape;
        std::vector<intptr_t> data_strides;
        intptr_t data_offset = 0; // The byte offset of the new histogram bin content array.
        size_t oaxis = 0; // The current axis of the original histogram.
        std::vector<intptr_t> const & histshape = bc_.get_shape_vector();
        std::vector<intptr_t> const & histstrides = bc_.get_data_strides_vector();
        for(size_t i=0; i<nsdim; ++i)
        {
            bp::object const item = bp::extract<bp::object>(seq[i]);
            if(detail::py::is_object_of_type(item, PyInt_Type))
            {
                // A single integer reduces the number of dimensions by one.
                // The previous dimensions get an offset
                // according to the given index (the integer). That means, one
                // has to jump over the index indices of the current dimension.
                intptr_t index = bp::extract<intptr_t>(item);
                index += (index < 0 ? histshape[oaxis] : 0);
                if(index < 0 || index >= histshape[oaxis])
                {
                    std::stringstream ss;
                    ss << "The index for axis "<<oaxis<<" must be in the "
                       << "interval [-"<<histshape[oaxis]<<", "
                       << histshape[oaxis]<<")!";
                    throw IndexError(ss.str());
                }
                data_offset += index*histstrides[oaxis];
                ++oaxis;
            }
            else if(detail::py::is_object_of_type(item, PySlice_Type))
            {
                // A slice object keeps the dimension. It changes its
                // stride if the step size is >1. Also it changes the offset,
                // if start >0 and it changes the shape of that dimension if
                // stop-start != histshape[oaxis].
                bp::slice sl = bp::extract<bp::slice>(item);

                // Construct the interval [start, stop) with jumps of size step.

                // Extract the step size.
                intptr_t step = 1;
                if(sl.step() != bp::object())
                {
                    step = bp::extract<intptr_t>(sl.step());
                }
                // We support only steps of +1 and -1. Otherwise the slicing
                // would result into non-contiguous histogram axes. Such axes
                // are not supported in this version of ndhist and probably will
                // never be.
                if(step != 1 && step != -1)
                {
                    std::stringstream ss;
                    ss << "The step of the slice for dimension "
                        << oaxis << " can only be 1 or -1!";
                    throw IndexError(ss.str());
                }

                // Setup the start index.
                intptr_t start = 0;
                if(sl.start() == bp::object())
                {
                    if(step < 0)
                    {
                        start = histshape[oaxis]-1;
                    }
                }
                else
                {
                    start = bp::extract<intptr_t>(sl.start());
                    start += (start < 0 ? histshape[oaxis] : 0);
                }
                if(start < 0 || start >= histshape[oaxis])
                {
                    std::stringstream ss;
                    ss << "The start index of the slice for dimension "
                        << oaxis << " must be in the interval ["
                        << "-"<<histshape[oaxis] << ", " << histshape[oaxis]
                        << ")!";
                    throw IndexError(ss.str());
                }

                // Setup the stop index.
                intptr_t stop = histshape[oaxis];
                if(sl.stop() == bp::object())
                {
                    if(step < 0)
                    {
                        stop = -1;
                    }
                }
                else
                {
                    stop = bp::extract<intptr_t>(sl.stop());
                    stop += (stop < 0 ? histshape[oaxis] : 0);
                }
                if(stop < -1 || stop > histshape[oaxis])
                {
                    std::stringstream ss;
                    ss << "The stop index of the slice for dimension "
                        << oaxis << " must be in the interval ["
                        << "-"<<histshape[oaxis]+1<<", "<<histshape[oaxis]
                        << ")!";
                    throw IndexError(ss.str());
                }

                intptr_t const dist = stop - start;
                if(dist == 0)
                {
                    std::stringstream ss;
                    ss << "The slice of dimension "<<oaxis<<" does not select "
                       << "any bin!";
                    throw IndexError(ss.str());
                }

                // Check if the step and distance sign match up.
                if((dist > 0) != (step > 0))
                {
                    std::stringstream ss;
                    ss << "The sign of step does not match up with the sign "
                       << "of stop-start for dimension "<<oaxis<<"!";
                    throw IndexError(ss.str());
                }

                // Setup shape, strides, and offset for this slice.
                intptr_t nbins = 0;
                if(dist < 0)
                {
                    nbins = -dist / -step + ((-dist % -step) != 0);
                }
                else
                {
                    nbins = dist / step + ((dist % step) != 0);
                }
                std::cout << "axis "<<oaxis<<": start = "<<start<<std::endl;
                std::cout << "axis "<<oaxis<<": stop = "<<stop<<std::endl;
                std::cout << "axis "<<oaxis<<": step = "<<step<<std::endl;
                std::cout << "axis "<<oaxis<<": nbins = "<<nbins<<std::endl;
                data_shape.push_back(nbins);

                // Setup the stride.
                data_strides.push_back(step*histstrides[oaxis]);

                // Setup the offset.
                data_offset += start * histstrides[oaxis];

                std::cout << "axis "<<oaxis<<": stride = "<<data_strides[data_strides.size()-1]<<std::endl;
                std::cout << "axis "<<oaxis<<": offset = "<<data_offset<<std::endl;

                axes.push_back(axes_[oaxis]->create_axis_slice(start, stop, step, nbins));

                ++oaxis;
            }
            else if(detail::py::is_object_of_type(item, PyEllipsis_Type))
            {
                // The ellipsis means that it should take the full range of
                // each dimension that follows, until the last dimensions are
                // specified explicitly.
                size_t const remaining_oaxes = nsdim - (i+1);
                size_t const n_axes = nd_ - oaxis - remaining_oaxes;
                for(size_t j=0; j<n_axes; ++j)
                {
                    // Add a full axis range for each axis.
                    data_shape.push_back(histshape[oaxis]);
                    data_strides.push_back(histstrides[oaxis]);

                    axes.push_back(axes_[oaxis]->create_axis_slice(0, histshape[oaxis], 1, histshape[oaxis]));

                    ++oaxis;
                }
            }
        }
        // Check if all dimensions have been adressed already. If not the
        // remaining dimensions are added with there full range.
        if(oaxis < nd_)
        {
            // Add the remaining axes as full ranges to the selection.
            size_t const n_axes = nd_ - oaxis;

            for(size_t j=0; j<n_axes; ++j)
            {
                data_shape.push_back(histshape[oaxis]);
                data_strides.push_back(histstrides[oaxis]);

                axes.push_back(axes_[oaxis]->create_axis_slice(0, histshape[oaxis], 1, histshape[oaxis]));

                std::cout << "Auto add axis "<<oaxis<<std::endl;
                ++oaxis;
            }
        }
        else if(oaxis > nd_)
        {
            std::stringstream ss;
            ss << "The internal axis counting failed. This is a BUG!";
            throw RuntimeError(ss.str());
        }

        // Create an ndhist object, is a data view into this ndhist object.
        std::cout << "data_offset = " << data_offset << std::endl;
        return ndhist(*this, axes, data_offset, data_shape, data_strides);
    }
    else
    {
        std::stringstream ss;
        ss << "The given slicing object does not initiate basic indexing "
           << "according to numpy basic indexing!";
        throw ValueError(ss.str());
    }
}

bool
ndhist::
is_compatible(ndhist const & other) const
{
    if(nd_ != other.nd_) {
        return false;
    }
    for(uintptr_t i=0; i<nd_; ++i)
    {
        bn::ndarray const this_axis_edges_arr = this->get_axes()[i]->get_binedges_ndarray();
        bn::ndarray const other_axis_edges_arr = other.get_axes()[i]->get_binedges_ndarray();

        if(this_axis_edges_arr.shape(0) != other_axis_edges_arr.shape(0))
        {
            return false;
        }

        if(bn::all(bn::equal(this_axis_edges_arr, other_axis_edges_arr), 0) == false)
        {
            return false;
        }
    }

    return true;
}

void
ndhist::
clear()
{
    clear_fct_(*this);
}

ndhist
ndhist::
empty_like() const
{
    bp::list axis_list;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        axis_list.append(axes_[i]->deepcopy());
    }
    bp::tuple axes(axis_list);
    return ndhist(axes, bc_weight_dt_, bc_class_);
}

ndhist
ndhist::
project(bp::object const & dims) const
{
    intptr_t const nd = get_nd();
    bn::ndarray axes_arr = bn::from_object(dims, bn::dtype::get_builtin<intptr_t>(), 0, 1, bn::ndarray::ALIGNED);
    std::set<intptr_t> axes;
    bn::iterators::flat_iterator< bn::iterators::single_value<intptr_t> > axes_arr_iter(axes_arr);
    while(! axes_arr_iter.is_end())
    {
        intptr_t axis = *axes_arr_iter;
        if(axis < 0) {
            axis += nd;
        }
        if(axis < 0)
        {
            std::stringstream ss;
            ss << "The axis value \""<< *axes_arr_iter <<"\" specifies an "
               << "axis < 0!";
            throw IndexError(ss.str());
        }
        else if(axis >= nd)
        {
            std::stringstream ss;
            ss << "The axis value \""<< axis <<"\" must be smaller than the "
               << "dimensionality of the histogram, i.e. smaller than "
               << nd <<"!";
            throw IndexError(ss.str());
        }
        if(! axes.insert(axis).second)
        {
            std::stringstream ss;
            ss << "The axis value \""<< axis <<"\" has been "
               << "specified at least twice!";
            throw ValueError(ss.str());
        }
        ++axes_arr_iter;
    }
    return project_fct_(*this, axes);
}

boost::shared_ptr<ndhist>
ndhist::
merge_axis_bins(
    intptr_t const axis
  , intptr_t const nbins_to_merge
  , bool const copy
)
{
    boost::shared_ptr<ndhist> self = this->shared_from_this();

    if(   nbins_to_merge == 0
       || nbins_to_merge == 1
      )
    {
        // We have nothing to do.
        return self;
    }

    // Make a deepcopy if this ndhist object is a data view into an other
    // ndhist object, or the user explicitly requested a copy.
    // Otherwise the rebin operation would invalidate the
    // original ndhist object.
    if(is_view() || copy)
    {
        self = this->deepcopy();
    }

    merge_axis_bins_fct_(*self, axis, nbins_to_merge);

    return self;
}

boost::shared_ptr<ndhist>
ndhist::
merge_bins(
    std::vector<intptr_t> const & axes
  , std::vector<intptr_t> const & nbins_to_merge
  , bool const copy
)
{
    boost::shared_ptr<ndhist> self = this->shared_from_this();

    if(axes.size() != nbins_to_merge.size())
    {
        std::stringstream ss;
        ss << "The lengths of the axes and nbins_to_merge tuples must be equal!";
        throw ValueError(ss.str());
    }
    bool work_to_do = false;
    for(size_t i=0; i<axes.size(); ++i)
    {
        if(nbins_to_merge[i] < 0)
        {
            std::stringstream ss;
            ss << "All nbins_to_merge values must be positive integer numbers!";
            throw ValueError(ss.str());
        }
        if(nbins_to_merge[i] >= 2)
        {
            work_to_do = true;
        }
    }
    if(! work_to_do)
    {
        return self;
    }

    // Make a deepcopy if this ndhist object is a data view into an other
    // ndhist object, or the user explicitly requested a copy.
    // Otherwise the rebin operation would invalidate the
    // original ndhist object.
    if(is_view() || copy)
    {
        self = this->deepcopy();
    }

    for(size_t i=0; i<axes.size(); ++i)
    {
        merge_axis_bins_fct_(*self, axes[i], nbins_to_merge[i]);
    }

    return self;
}

boost::shared_ptr<ndhist>
ndhist::
merge_bins(
    bp::tuple const & axes
  , bp::tuple const & nbins_to_merge
  , bool const copy
)
{
    std::vector<intptr_t> axes_vec(bp::len(axes));
    for(size_t i=0; i<axes_vec.size(); ++i)
    {
        axes_vec[i] = bp::extract<intptr_t>(axes[i]);
    }
    std::vector<intptr_t> nbins_to_merge_vec(bp::len(nbins_to_merge));
    for(size_t i=0; i<nbins_to_merge_vec.size(); ++i)
    {
        nbins_to_merge_vec[i] = bp::extract<intptr_t>(nbins_to_merge[i]);
    }

    return merge_bins(axes_vec, nbins_to_merge_vec, copy);
}

bp::tuple
ndhist::
py_get_shape() const
{
    std::vector<intptr_t> shape(nd_);
    for(uintptr_t i=0; i<nd_; ++i)
    {
        shape[i] = axes_[i]->get_n_bins();
    }
    return boost::python::make_tuple_from_container(shape.begin(), shape.end());
}

std::vector<intptr_t>
ndhist::
get_nbins() const
{
    std::vector<intptr_t> nbins(nd_);
    for(uintptr_t i=0; i<nd_; ++i)
    {
        nbins[i] = axes_[i]->get_n_bins();
        if(axes_[i]->has_underflow_bin()) --(nbins[i]);
        if(axes_[i]->has_overflow_bin()) --(nbins[i]);
    }
    return nbins;
}

bp::tuple
ndhist::
py_get_nbins() const
{
    std::vector<intptr_t> nbins = get_nbins();
    return boost::python::make_tuple_from_container(nbins.begin(), nbins.end());
}

bp::tuple
ndhist::
py_get_axes() const
{
    return boost::python::make_tuple_from_container(axes_.begin(), axes_.end());
}

bp::object
ndhist::
py_get_noe_ndarray() const
{
    // The core part of the bin content array excludes the under- and
    // overflow bins. So we need to create an appropriate view into the bin
    // content array.
    std::vector<intptr_t> shape;
    std::vector<intptr_t> front_capacity;
    std::vector<intptr_t> back_capacity;
    calc_core_bin_content_ndarray_settings(shape, front_capacity, back_capacity);

    intptr_t const sub_item_byte_offset = 0;

    bn::ndarray arr = detail::ndarray_storage::construct_ndarray(bc_, bc_noe_dt_, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
    if(nd_ == 0)
    {
        return arr.scalarize();
    }
    return arr;
}

bp::object
ndhist::
py_get_sow_ndarray() const
{
    // The core part of the bin content array excludes the under- and
    // overflow bins. So we need to create an appropriate view into the bin
    // content array.
    std::vector<intptr_t> shape;
    std::vector<intptr_t> front_capacity;
    std::vector<intptr_t> back_capacity;
    calc_core_bin_content_ndarray_settings(shape, front_capacity, back_capacity);

    intptr_t const sub_item_byte_offset = bc_.get_dtype().get_fields_byte_offsets()[1];

    bn::ndarray arr = detail::ndarray_storage::construct_ndarray(bc_, bc_weight_dt_, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
    if(nd_ == 0)
    {
        return arr.scalarize();
    }
    return arr;
}

bp::object
ndhist::
py_get_sows_ndarray() const
{
    // The core part of the bin content array excludes the under- and
    // overflow bins. So we need to create an appropriate view into the bin
    // content array.
    std::vector<intptr_t> shape;
    std::vector<intptr_t> front_capacity;
    std::vector<intptr_t> back_capacity;
    calc_core_bin_content_ndarray_settings(shape, front_capacity, back_capacity);

    intptr_t const sub_item_byte_offset = bc_.get_dtype().get_fields_byte_offsets()[2];

    bn::ndarray arr = detail::ndarray_storage::construct_ndarray(bc_, bc_weight_dt_, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
    if(nd_ == 0)
    {
        return arr.scalarize();
    }
    return arr;
}

bp::tuple
ndhist::
py_get_labels() const
{
    std::vector<std::string> labels;
    labels.reserve(nd_);
    for(uintptr_t i=0; i<nd_; ++i)
    {
        labels.push_back(axes_[i]->get_label());
    }
    return boost::python::make_tuple_from_container(labels.begin(), labels.end());
}

bp::tuple
ndhist::
py_get_underflow_entries() const
{
    std::vector<bn::ndarray> arrays = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 0);

    for(size_t i=0; i<arrays.size(); ++i)
    {
        arrays[i] = arrays[i].deepcopy();
    }

    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_underflow_entries_view() const
{
    std::vector<bn::ndarray> arrays = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 0);
    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_overflow_entries() const
{
    std::vector<bn::ndarray> arrays = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 0);

    for(size_t i=0; i<arrays.size(); ++i)
    {
        arrays[i] = arrays[i].deepcopy();
    }

    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_overflow_entries_view() const
{
    std::vector<bn::ndarray> arrays = this->get_noe_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 0);
    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_underflow() const
{
    std::vector<bn::ndarray> arrays = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 1);

    for(size_t i=0; i<arrays.size(); ++i)
    {
        arrays[i] = arrays[i].deepcopy();
    }

    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_underflow_view() const
{
    std::vector<bn::ndarray> arrays = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 1);
    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_overflow() const
{
    std::vector<bn::ndarray> arrays = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 1);

    for(size_t i=0; i<arrays.size(); ++i)
    {
        arrays[i] = arrays[i].deepcopy();
    }

    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_overflow_view() const
{
    std::vector<bn::ndarray> arrays = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 1);
    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_underflow_squaredweights() const
{
    std::vector<bn::ndarray> arrays = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 2);

    for(size_t i=0; i<arrays.size(); ++i)
    {
        arrays[i] = arrays[i].deepcopy();
    }

    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_underflow_squaredweights_view() const
{
    std::vector<bn::ndarray> arrays = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_UNDERFLOW, 2);
    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_overflow_squaredweights() const
{
    std::vector<bn::ndarray> arrays = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 2);

    for(size_t i=0; i<arrays.size(); ++i)
    {
        arrays[i] = arrays[i].deepcopy();
    }

    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bp::tuple
ndhist::
py_get_overflow_squaredweights_view() const
{
    std::vector<bn::ndarray> arrays = this->get_weight_type_field_axes_oor_ndarrays_fct_(*this, axis::OOR_OVERFLOW, 2);
    return boost::python::make_tuple_from_container(arrays.begin(), arrays.end());
}

bn::ndarray
ndhist::
get_binedges_ndarray(intptr_t axis) const
{
    // Count axis from the back if axis is negative.
    if(axis < 0) {
        axis += axes_.size();
    }

    if(axis < 0 || axis >= intptr_t(axes_.size()))
    {
        std::stringstream ss;
        ss << "The axis parameter must be in the interval "
           << "[0, " << axes_.size()-1 << "] or "
           << "[-1, -"<< axes_.size() <<"]!";
        throw IndexError(ss.str());
    }

    // If the axis has oor bins, we need to cut them and return the cutted
    // result.
    if(axes_[axis]->has_underflow_bin() && axes_[axis]->has_overflow_bin())
    {
        bn::ndarray const allbinedges = axes_[axis]->get_binedges_ndarray();
        bp::slice sl(1, -1, 1);
        return allbinedges[sl];
    }
    else if(axes_[axis]->has_underflow_bin())
    {
        bn::ndarray const allbinedges = axes_[axis]->get_binedges_ndarray();
        bp::slice sl(1, axes_[axis]->get_n_bins(), 1);
        return allbinedges[sl];
    }
    else if(axes_[axis]->has_overflow_bin())
    {
        bn::ndarray const allbinedges = axes_[axis]->get_binedges_ndarray();
        bp::slice sl(0, -1, 1);
        return allbinedges[sl];
    }

    return axes_[axis]->get_binedges_ndarray();
}

bp::object
ndhist::
py_get_binedges() const
{
    // Special case for 1d histograms, where we skip the extra tuple of
    // length 1.
    if(nd_ == 1)
    {
        return get_binedges_ndarray(0);
    }

    std::vector<bn::ndarray> vec;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        vec.push_back(get_binedges_ndarray(i));
    }
    return boost::python::make_tuple_from_container(vec.begin(), vec.end());
}

bn::ndarray
ndhist::
get_bincenters_ndarray(intptr_t axis) const
{
    // Count axis from the back if axis is negative.
    if(axis < 0) {
        axis += axes_.size();
    }

    if(axis < 0 || axis >= intptr_t(axes_.size()))
    {
        std::stringstream ss;
        ss << "The axis parameter must be in the interval "
           << "[0, " << axes_.size()-1 << "] or "
           << "[-1, -"<< axes_.size() <<"]!";
        throw IndexError(ss.str());
    }

    // If the axis has oor bins, we need to cut them and return the cutted
    // result.
    if(axes_[axis]->has_underflow_bin() && axes_[axis]->has_overflow_bin())
    {
        bn::ndarray const allbincenters = axes_[axis]->get_bincenters_ndarray();
        bp::slice sl(1, -1, 1);
        return allbincenters[sl];
    }
    else if(axes_[axis]->has_underflow_bin())
    {
        bn::ndarray const allbincenters = axes_[axis]->get_bincenters_ndarray();
        bp::slice sl(1, axes_[axis]->get_n_bins(), 1);
        return allbincenters[sl];
    }
    else if(axes_[axis]->has_overflow_bin())
    {
        bn::ndarray const allbincenters = axes_[axis]->get_bincenters_ndarray();
        bp::slice sl(0, -1, 1);
        return allbincenters[sl];
    }

    return axes_[axis]->get_bincenters_ndarray();
}

bp::object
ndhist::
py_get_bincenters() const
{
    // Special case for 1d histograms, where we skip the extra tuple of
    // length 1.
    if(nd_ == 1)
    {
        return get_bincenters_ndarray(0);
    }

    std::vector<bn::ndarray> vec;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        vec.push_back(get_bincenters_ndarray(i));
    }
    return boost::python::make_tuple_from_container(vec.begin(), vec.end());
}

bn::ndarray
ndhist::
get_binwidths_ndarray(intptr_t axis) const
{
    // Count axis from the back if axis is negative.
    if(axis < 0) {
        axis += axes_.size();
    }

    if(axis < 0 || axis >= intptr_t(axes_.size()))
    {
        std::stringstream ss;
        ss << "The axis parameter must be in the interval "
           << "[0, " << axes_.size()-1 << "] or "
           << "[-1, -"<< axes_.size() <<"]!";
        throw IndexError(ss.str());
    }

    // If the axis has oor bins, we need to cut them and return the cutted
    // result.
    if(axes_[axis]->has_underflow_bin() && axes_[axis]->has_overflow_bin())
    {
        bn::ndarray const allbinwidths = axes_[axis]->get_binwidths_ndarray();
        bp::slice sl(1, -1, 1);
        return allbinwidths[sl];
    }
    else if(axes_[axis]->has_underflow_bin())
    {
        bn::ndarray const allbinwidths = axes_[axis]->get_binwidths_ndarray();
        bp::slice sl(1, axes_[axis]->get_n_bins(), 1);
        return allbinwidths[sl];
    }
    else if(axes_[axis]->has_overflow_bin())
    {
        bn::ndarray const allbinwidths = axes_[axis]->get_binwidths_ndarray();
        bp::slice sl(0, -1, 1);
        return allbinwidths[sl];
    }

    return axes_[axis]->get_binwidths_ndarray();
}

bp::object
ndhist::
py_get_binwidths() const
{
    // Special case for 1d histograms, where we skip the extra tuple of
    // length 1.
    if(nd_ == 1)
    {
        return get_binwidths_ndarray(0);
    }

    std::vector<bn::ndarray> vec;
    for(uintptr_t i=0; i<nd_; ++i)
    {
        vec.push_back(get_binwidths_ndarray(i));
    }
    return boost::python::make_tuple_from_container(vec.begin(), vec.end());
}

void
ndhist::
py_fill(bp::object const & ndvalue_obj, bp::object weight_obj)
{
    // In case None is given as weight, we will use one.
    if(weight_obj == bp::object())
    {
        weight_obj = bp::object(1);
    }
    fill_fct_(*this, ndvalue_obj, weight_obj);
}

// TODO: Use the new multi_axis_iter for this.
static
void
initialize_extended_array_axis_range(
    bn::iterators::flat_iterator< bn::iterators::single_value<bp::object> > & iter
  , intptr_t axis
  , std::vector<intptr_t> const & shape
  , std::vector<intptr_t> const & strides
  , intptr_t n_iters
  , intptr_t axis_idx_range_min
  , intptr_t axis_idx_range_max
  , bp::object const obj_class
)
{
    int const nd = strides.size();

    intptr_t const last_axis = (nd - 1 == axis ? nd - 2 : nd - 1);
    //std::cout << "last_axis = "<< last_axis << std::endl<<std::flush;
    std::vector<intptr_t> indices(nd);
    for(intptr_t axis_idx=axis_idx_range_min; axis_idx < axis_idx_range_max; ++axis_idx)
    {
        //std::cout << "Start new axis idx"<<std::endl<<std::flush;
        memset(&indices.front(), 0, nd*sizeof(intptr_t));
        indices[axis] = axis_idx;
        // The iteration follows a matrix. The index pointer p indicates
        // index that needs to be incremented.
        // We need to start from the innermost dimension, unless it is the
        // iteration axis.
        intptr_t p = last_axis;
        for(intptr_t i=0; i<n_iters; ++i)
        {
            //std::cout << "indices = ";
            //for(intptr_t j=0; j<nd; ++j)
            //{
                //std::cout << indices[j] << ",";
            //}
            //std::cout << std::endl;

            intptr_t iteridx = 0;
            for(intptr_t j=nd-1; j>=0; --j)
            {
                iteridx += indices[j]*strides[j];
            }
            //std::cout << "iteridx = " << iteridx << std::endl<<std::flush;
            iter.jump_to_iter_index(iteridx);
            //std::cout << "jump done" << std::endl<<std::flush;

            *iter;
            uintptr_t * obj_ptr_ptr = iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            //std::cout << "Setting pointer data ..."<<std::flush;
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
            //std::cout << "done."<<std::endl<<std::flush;
            if(i == n_iters-1) break;
            // Move the index pointer to the next outer-axis if the index
            // of the current axis has reached its maximum. Then increase
            // the index and reset all indices to the right of this
            // increased index to zero. After this operation, the index
            // pointer points to the inner-most axis (excluding the
            // iteration axis).
            //std::cout << "p1 = "<<p<<std::endl<<std::flush;
            while(indices[p] == shape[p]-1)
            {
                --p;
                if(p == axis) --p;
            }
            //std::cout << "p2 = "<<p<<std::endl<<std::flush;
            indices[p]++;
            while(p < last_axis)
            {
                ++p;
                if(p == axis) ++p;
                indices[p] = 0;
            }

            //std::cout << "p3 = "<<p<<std::endl;
        }
    }
}

void
ndhist::
initialize_extended_array_axis(
    bp::object & arr_obj
  , bp::object const & obj_class
  , intptr_t axis
  , intptr_t f_n_extra_bins
  , intptr_t b_n_extra_bins
)
{
    if(f_n_extra_bins == 0 && b_n_extra_bins == 0) return;

    bn::ndarray & arr = *static_cast<bn::ndarray *>(&arr_obj);
    std::vector<intptr_t> const shape = arr.get_shape_vector();

    intptr_t f_axis_idx_range_min = 0;
    intptr_t f_axis_idx_range_max = 0;
    if(f_n_extra_bins > 0)
    {
        f_axis_idx_range_min = 0;
        f_axis_idx_range_max = f_n_extra_bins;
    }

    intptr_t b_axis_idx_range_min = 0;
    intptr_t b_axis_idx_range_max = 0;
    if(b_n_extra_bins > 0)
    {
        b_axis_idx_range_min = shape[axis] - b_n_extra_bins;
        b_axis_idx_range_max = shape[axis];
    }

    bn::iterators::flat_iterator< bn::iterators::single_value<bp::object> > bc_iter(arr, bn::detail::iter_operand::flags::WRITEONLY::value);
    intptr_t const nd = arr.get_nd();
    if(nd == 1)
    {
        // We can just use the flat iterator directly.

        // --- for front elements.
        for(intptr_t axis_idx = f_axis_idx_range_min; axis_idx < f_axis_idx_range_max; ++axis_idx)
        {
            bc_iter.jump_to_iter_index(axis_idx);
            *bc_iter;
            uintptr_t * obj_ptr_ptr = bc_iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }

        // --- for back elements.
        for(intptr_t axis_idx = b_axis_idx_range_min; axis_idx < b_axis_idx_range_max; ++axis_idx)
        {
            bc_iter.jump_to_iter_index(axis_idx);
            *bc_iter;
            uintptr_t * obj_ptr_ptr = bc_iter.get_value_type_traits().value_obj_ptr_;
            bp::object obj = obj_class();
            *obj_ptr_ptr = reinterpret_cast<uintptr_t>(bp::incref<PyObject>(obj.ptr()));
        }
    }
    else
    {
        // Create a strides vector for the iteration.
        std::vector<intptr_t> strides(nd);
        strides[nd-1] = 1;
        for(intptr_t i=nd-2; i>=0; --i)
        {
            strides[i] = shape[i+1]*strides[i+1];
        }

        // Calculate the number of iterations (without the iteration axis).
        //std::cout << "shape = ";
        intptr_t n_iters = 1;
        for(intptr_t i=0; i<nd; ++i)
        {
            //std::cout << shape[i] << ",";
            if(i != axis) {
                n_iters *= shape[i];
            }
        }
        //std::cout << std::endl;
        //std::cout << "n_iters = " << n_iters << std::endl<<std::flush;

        // Initialize the front elements.
        initialize_extended_array_axis_range(bc_iter, axis, shape, strides, n_iters, f_axis_idx_range_min, f_axis_idx_range_max, obj_class);
        // Initialize the back elements.
        initialize_extended_array_axis_range(bc_iter, axis, shape, strides, n_iters, b_axis_idx_range_min, b_axis_idx_range_max, obj_class);
    }
}

void
ndhist::
extend_axes(
    std::vector<intptr_t> const & f_n_extra_bins_vec
  , std::vector<intptr_t> const & b_n_extra_bins_vec
)
{
    for(uintptr_t i=0; i<nd_; ++i)
    {
        Axis & axis = *this->axes_[i];
        if(axis.is_extendable())
        {
            axis.extend(f_n_extra_bins_vec[i], b_n_extra_bins_vec[i]);
        }
    }
}

void
ndhist::
extend_bin_content_array(
    std::vector<intptr_t> const & f_n_extra_bins_vec
  , std::vector<intptr_t> const & b_n_extra_bins_vec
)
{
    // Extend the bin content array. This might cause a reallocation of memory.
    bc_.extend_axes(f_n_extra_bins_vec, b_n_extra_bins_vec, axes_extension_max_fcap_vec_, axes_extension_max_bcap_vec_);

    // We need to initialize the new bin content values, if the data type
    // is object.
    if(! bn::dtype::equivalent(bc_weight_dt_, bn::dtype::get_builtin<bp::object>()))
        return;

    std::vector<intptr_t> f_n_extra_bins = f_n_extra_bins_vec;
    std::vector<intptr_t> b_n_extra_bins = b_n_extra_bins_vec;

    bn::ndarray bc_sow_arr  = construct_complete_bin_content_ndarray(bc_weight_dt_, 1);
    bn::ndarray bc_sows_arr = construct_complete_bin_content_ndarray(bc_weight_dt_, 2);
    for(uintptr_t axis=0; axis<nd_; ++axis)
    {
        // In case the axis is extendable we also need to initialize the shifted
        // virtual under and overflow bin.
        if(axes_[axis]->is_extendable())
        {
            if(f_n_extra_bins[axis] > 0) { f_n_extra_bins[axis] += 1; }
            if(b_n_extra_bins[axis] > 0) { b_n_extra_bins[axis] += 1; }
        }
        initialize_extended_array_axis(bc_sow_arr,  bc_class_, axis, f_n_extra_bins[axis], b_n_extra_bins[axis]);
        initialize_extended_array_axis(bc_sows_arr, bc_class_, axis, f_n_extra_bins[axis], b_n_extra_bins[axis]);
    }
}

bn::ndarray
ndhist::
construct_complete_bin_content_ndarray(
    bn::dtype const & dt
  , size_t const field_idx
) const
{
    std::vector<intptr_t> shape          = bc_.get_shape_vector();
    std::vector<intptr_t> front_capacity = bc_.get_front_capacity_vector();
    std::vector<intptr_t> back_capacity  = bc_.get_back_capacity_vector();
    // Add the under- and overflow bins of the extendable axes to the shape, and
    // remove them from the front- and back capacities, in order to calculate
    // the data offset and strides correctly.
    for(uintptr_t i=0; i<nd_; ++i)
    {
        if(axes_[i]->is_extendable())
        {
            shape[i] += 2;
            front_capacity[i] -= 1;
            back_capacity[i] -= 1;
        }
    }

    intptr_t const sub_item_byte_offset = (field_idx == 0 ? 0 : bc_.get_dtype().get_fields_byte_offsets()[field_idx]);

    return detail::ndarray_storage::construct_ndarray(bc_, dt, shape, front_capacity, back_capacity, sub_item_byte_offset, /*owner=*/NULL, /*set_owndata_flag=*/false);
}

}//namespace ndhist
