import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_ndhist_basic_slicing(self):
        """Tests if the basic slicing a la numpy works properly.

        """
        axis_0 = ndhist.axes.linear(-1, 3, 1)

        h = ndhist.ndhist((axis_0,))
        self.assertFalse(h.is_view)
        self.assertTrue(h.nbins == (4,))

        # Fill into underflow bin.
        h.fill((-1.1,), 2.3)
        # Fill into bin [0].
        h.fill((-0.6,), 2.4)
        # Fill into bin [1].
        h.fill((0.3,), 2.5)
        # Fill into bin [2].
        h.fill((1.3,), 2.6)
        # Fill into bin [3].
        h.fill((2,), 2.7)
        # Fill into overflow bin.
        h.fill((3.1,), 2.8)
        self.assertTrue(h[0].bincontent == 2.3)
        self.assertTrue(h.bincontent[0] == 2.4)
        self.assertTrue(h.bincontent[1] == 2.5)
        self.assertTrue(h.bincontent[2] == 2.6)
        self.assertTrue(h.bincontent[3] == 2.7)
        self.assertTrue(h[5].bincontent == 2.8)

        h1 = h[0:6]
        self.assertTrue(h1.is_view)
        self.assertTrue(h1.nbins == (4,))
        self.assertTrue(h1.axes[0].has_underflow_bin)
        self.assertTrue(h1.axes[0].has_overflow_bin)

        h2 = h[1:5]
        self.assertTrue(h2.is_view)
        self.assertTrue(h2.nbins == (4,))
        self.assertFalse(h2.axes[0].has_underflow_bin)
        self.assertFalse(h2.axes[0].has_overflow_bin)

        h3 = h[0:1]
        self.assertTrue(h3.is_view)
        self.assertTrue(h3.nbins == (0,))
        self.assertTrue(h3.axes[0].has_underflow_bin)
        self.assertFalse(h3.axes[0].has_overflow_bin)
        self.assertTrue(h3.bincontent.size == 0)
        self.assertTrue(h3[0].bincontent == 2.3)

        # Integers reduce the dimensionality.
        h4 = h[0]
        self.assertTrue(h4.is_view)
        self.assertTrue(len(h4.axes) == 0)
        self.assertTrue(h4.ndim == 0)
        self.assertTrue(h4.bincontent == 2.3)

        # A slice object keeps the dimensionality.
        h5 = h[2:5]
        self.assertTrue(h5.is_view)
        self.assertFalse(h5.axes[0].has_underflow_bin)
        self.assertFalse(h5.axes[0].has_overflow_bin)
        self.assertTrue(h5.ndim == 1)

        # An ellipsis keeps also the dimensionality.
        h6 = h[...]
        self.assertTrue(h6.is_view)
        self.assertTrue(h6.axes[0].has_underflow_bin)
        self.assertTrue(h6.axes[0].has_overflow_bin)
        self.assertTrue(h6.ndim == 1)
        self.assertTrue(h6.shape == (6,))
        self.assertTrue(h6.nbins == (4,))
        self.assertTrue(np.all(h6.axes[0].binedges == h.axes[0].binedges))

if(__name__ == "__main__"):
    unittest.main()
