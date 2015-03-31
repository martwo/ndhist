import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_ndhist_rebin_axis_method(self):
        """Tests if the rebin_axis method of the ndhist class works.

        """
        axis_0 = ndhist.axes.linear(0,10)

        h = ndhist.ndhist((axis_0,))

        h.fill(([0,1,2,3,4,5,6,7,8,9,10,10,10],))

        self.assertTrue(np.all(h.bincontent == 1))

        h2 = h.rebin_axis(0, 3)

if(__name__ == "__main__"):
    unittest.main()
