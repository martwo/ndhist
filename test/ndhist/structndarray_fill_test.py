import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_struct_ndarray_fill(self):
        """Tests of the fill method works when providing a structured ndarray
        holding the values for each dimension for each entry.

        """
        axis_0 = ndhist.axes.linear(-2, 3, 1)
        axis_1 = ndhist.axes.linear(-1, 2, 1)
        self.assertTrue(axis_0.nbins == 7)
        self.assertTrue(axis_1.nbins == 5)

        h = ndhist.ndhist((axis_0,axis_1))
        self.assertTrue(np.any(h.binentries) == False)
        self.assertTrue(np.any(h.bincontent) == False)

        ndvalues = np.empty(3, dtype=[(h.axes[0].name, h.axes[0].dtype), (h.axes[1].name, h.axes[1].dtype)])
        ndvalues[h.axes[0].name] = np.array([-1, -0.5, 0])
        ndvalues[h.axes[1].name] = np.array([ 1,  1.1, 0])

        weights = np.array([1, 2, 1])

        h.fill(ndvalues, weights)

        self.assertTrue(h.binentries[1,2] == 2)
        self.assertTrue(h.bincontent[1,2] == 3)
        self.assertTrue(h.squaredweights[1,2] == 5)

        self.assertTrue(h.binentries[2,1] == 1)
        self.assertTrue(h.bincontent[2,1] == 1)
        self.assertTrue(h.squaredweights[2,1] == 1)

if(__name__ == "__main__"):
    unittest.main()
