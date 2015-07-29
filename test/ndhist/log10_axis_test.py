import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_log10_axis_1D(self):
        """Tests if the log10_axis works with the ndhist object for 1D
        histograms.

        """
        axis_0 = ndhist.axes.log10(0.1, 100, 0.1)
        self.assertTrue(axis_0.nbins == 32)

        h = ndhist.ndhist((axis_0,))
        self.assertTrue(np.any(h.binentries) == False)
        self.assertTrue(np.any(h.bincontent) == False)

        h.fill([0.1, 0.2, 99.])
        self.assertTrue(np.all(h.bincontent == np.array([
            1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  1.])))

if(__name__ == "__main__"):
    unittest.main()
