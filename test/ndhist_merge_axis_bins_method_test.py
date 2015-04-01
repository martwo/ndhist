import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_ndhist_merge_axis_bins_method_1D(self):
        """Tests if the merge_axis_bins method of the ndhist class works
        properly for 1D histograms.

        """
        axis_0 = ndhist.axes.linear(0,10)

        h = ndhist.ndhist((axis_0,))
        self.assertTrue(h.axes[0].has_underflow_bin)
        self.assertTrue(h.axes[0].has_overflow_bin)
        self.assertTrue(h.shape == (12,))
        self.assertTrue(h.nbins == (10,))
        self.assertTrue(np.all(h.axes[0].binedges == np.array([-np.inf,0,1,2,3,4,5,6,7,8,9,10,+np.inf])))

        h.fill(([0,1,2,3,4,5,6,7,8,9,10,10,10],))
        self.assertTrue(np.all(h.bincontent == 1))
        self.assertTrue(np.all(h.overflow[0] == 3))

        h2 = h.merge_axis_bins(0, 3, copy=True)
        self.assertFalse(h2.is_view)
        self.assertTrue(h2.shape == (5,))
        self.assertTrue(h2.nbins == (3,))
        self.assertTrue(np.all(h2.bincontent == 3))
        self.assertTrue(h2.overflow[0] == 4)

    def test_ndhist_merge_axis_bins_method_1D_extendable_axis(self):
        """Tests if the merge_axis_bins method of the ndhist class works
        properly for 1D histograms using an extendable axis.

        """
        axis_0 = ndhist.axes.linear(0,10, extend=True, extracap=10)

        h = ndhist.ndhist((axis_0,))
        self.assertTrue(h.axes[0].is_extendable)
        self.assertFalse(h.axes[0].has_underflow_bin)
        self.assertFalse(h.axes[0].has_overflow_bin)
        self.assertTrue(h.shape == (10,))
        self.assertTrue(h.nbins == (10,))
        self.assertTrue(np.all(h.axes[0].binedges == np.array([0,1,2,3,4,5,6,7,8,9,10])))

        h.fill(([0,1,2,3,4,5,6,7,8,9,10,10,10],))
        self.assertTrue(h.shape == (11,))
        self.assertTrue(h.nbins == (11,))
        self.assertTrue(np.all(h.axes[0].binedges == np.array([0,1,2,3,4,5,6,7,8,9,10,11])))
        self.assertTrue(np.all(h.bincontent[:10] == 1))
        self.assertTrue(np.all(h.bincontent[10:11] == 3))

        h2 = h.merge_axis_bins(0, 3, copy=True)
        self.assertTrue(h2.axes[0].is_extendable)
        self.assertFalse(h2.axes[0].has_underflow_bin)
        self.assertFalse(h2.axes[0].has_overflow_bin)
        self.assertTrue(h2.shape == (3,))
        self.assertTrue(h2.nbins == (3,))
        self.assertTrue(np.all(h2.bincontent == 3))
        # Since it is an extendable axis, the remaining bins get discarded.
        self.assertTrue(h2.overflow[0] == 0)

    def test_ndhist_merge_axis_bins_method_1D_new_overflow_bin(self):
        """Tests if the merge_axis_bins method of the ndhist class works
        properly for 1D histograms, when the initial axis does not have an
        overflow bin, but the bin merging would result into a newly created
        overflow bin.

        """
        axis_0 = ndhist.axes.linear(0,10, addoorbins=False)

        h = ndhist.ndhist((axis_0,))
        self.assertFalse(h.axes[0].has_underflow_bin)
        self.assertFalse(h.axes[0].has_overflow_bin)
        self.assertTrue(h.shape == (10,))
        self.assertTrue(h.nbins == (10,))
        self.assertTrue(np.all(h.axes[0].binedges == np.array([0,1,2,3,4,5,6,7,8,9,10])))

        h.fill(([0,1,2,3,4,5,6,7,8,9,10,10,10],))
        # Since there is no overflow bin, the values 10 will be discarded.
        self.assertTrue(np.all(h.bincontent == 1))

        h2 = h.merge_axis_bins(0, 3, copy=True)
        self.assertFalse(h2.is_view)
        self.assertFalse(h2.axes[0].has_underflow_bin)
        self.assertTrue(h2.axes[0].has_overflow_bin)
        self.assertTrue(h2.shape == (4,))
        self.assertTrue(h2.nbins == (3,))
        self.assertTrue(np.all(h2.bincontent == 3))
        self.assertTrue(h2.overflow[0] == 1)
        self.assertTrue(np.all(h2.axes[0].binedges == np.array([0,3,6,9,10])))

    def test_ndhist_merge_axis_bins_method_2D(self):
        """Tests if the merge_axis_bins method of the ndhist class works
        properly for 2D histograms.

        """
        axis_0 = ndhist.axes.linear(0,10)
        axis_1 = ndhist.axes.linear(0,5)

        h = ndhist.ndhist((axis_0,axis_1))
        self.assertTrue(h.shape == (12,7))
        self.assertTrue(h.nbins == (10,5))

        h.fill(([0,1,2,3,4,5,6, 7,   8,9,10,10,10],
                [0,1,2,3,4,4,5,-1,-0.2,5, 6, 0, 1]))

        self.assertTrue(np.all(h.bincontent == np.array(
            [[ 1.,  0.,  0.,  0.,  0.],
             [ 0.,  1.,  0.,  0.,  0.],
             [ 0.,  0.,  1.,  0.,  0.],
             [ 0.,  0.,  0.,  1.,  0.],
             [ 0.,  0.,  0.,  0.,  1.],
             [ 0.,  0.,  0.,  0.,  1.],
             [ 0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.]])))
        self.assertTrue(np.all(h.underflow[0] == np.array(
            [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]
        )))
        self.assertTrue(np.all(h.underflow[1] == np.array(
            [[ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 1.],
             [ 1.],
             [ 0.],
             [ 0.]]
        )))
        self.assertTrue(np.all(h.overflow[0] == np.array(
            [[ 0.,  1.,  1.,  0.,  0.,  0.,  1.]]
        )))
        self.assertTrue(np.all(h.overflow[1] == np.array(
            [[ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 0.],
             [ 1.],
             [ 0.],
             [ 0.],
             [ 1.],
             [ 1.]]
        )))

        h2 = h.merge_axis_bins(0, 3, copy=True)
        self.assertFalse(h2.is_view)
        self.assertTrue(h2.shape == (5,7))
        self.assertTrue(h2.nbins == (3,5))

        self.assertTrue(np.all(h2.bincontent == np.array(
            [[ 1.,  1.,  1.,  0.,  0.],
             [ 0.,  0.,  0.,  1.,  2.],
             [ 0.,  0.,  0.,  0.,  0.]]
        )))
        self.assertTrue(np.all(h2.underflow[0] == np.array(
            [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]
        )))
        self.assertTrue(np.all(h2.underflow[1] == np.array(
            [[ 0.],
             [ 0.],
             [ 0.],
             [ 2.],
             [ 0.]]
        )))
        self.assertTrue(np.all(h2.overflow[0] == np.array(
            [[ 0.,  1.,  1.,  0.,  0.,  0.,  2.]]
        )))
        self.assertTrue(np.all(h2.overflow[1] == np.array(
            [[ 0.],
             [ 0.],
             [ 0.],
             [ 1.],
             [ 2.]]
        )))

if(__name__ == "__main__"):
    unittest.main()
