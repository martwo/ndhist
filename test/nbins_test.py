import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_nbins(self):
        # Create an axis with 3 bins (including the under- and overflow bins).
        axis_0 = ndhist.axes.constant_bin_width_axis(np.array([0.,1.,2.,3.]))
        self.assertTrue(axis_0.nbins == 3)

        h = ndhist.ndhist((axis_0,))
        self.assertTrue(len(h.nbins) == 1)
        self.assertTrue(h.nbins[0] == 1)

if(__name__ == "__main__"):
    unittest.main()
