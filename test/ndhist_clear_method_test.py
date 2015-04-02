import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_ndhist_clear_method(self):
        """Tests if the clear method of the ndhist class works properly.

        """
        axis_0 = ndhist.axes.linear(0,10)

        h = ndhist.ndhist((axis_0,))
        self.assertTrue(h.axes[0].has_underflow_bin)
        self.assertTrue(h.axes[0].has_overflow_bin)
        self.assertTrue(h.shape == (12,))
        self.assertTrue(h.nbins == (10,))
        self.assertTrue(np.all(h.axes[0].binedges == np.array(
            [-np.inf,0,1,2,3,4,5,6,7,8,9,10,+np.inf]
        )))

        h.fill(([0,1,2,3,4,5,6,7,8,9,10,10,10],))
        self.assertTrue(np.all(h.bincontent == 1))

        h2 = h[3:6]
        self.assertFalse(h2.axes[0].has_underflow_bin)
        self.assertFalse(h2.axes[0].has_overflow_bin)
        self.assertTrue(h2.shape == (3,))
        self.assertTrue(h2.nbins == (3,))
        self.assertTrue(np.all(h2.axes[0].binedges == np.array(
            [2., 3., 4., 5.]
        )))
        self.assertTrue(np.all(h2.bincontent == 1))

        h2.clear()
        self.assertTrue(np.all(h2.bincontent == 0))
        self.assertTrue(np.all(h.bincontent == np.array(
            [ 1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.]
        )))

if(__name__ == "__main__"):
    unittest.main()
