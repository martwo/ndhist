import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_tuple_fill_not_extendable_axes(self):
        """Tests if the fill method works when providing the ndvalues as a tuple
        with a ndarray for each dimension and all the axes are not extendable.

        """
        axis_0 = ndhist.axes.linear(-2, 3, 1)
        axis_1 = ndhist.axes.linear(-1, 2, 1)
        self.assertTrue(axis_0.nbins == 7)
        self.assertTrue(axis_1.nbins == 5)

        h = ndhist.ndhist((axis_0,axis_1))
        self.assertTrue(np.any(h.binentries) == False)
        self.assertTrue(np.any(h.bincontent) == False)

        h.fill((-1, 1))
        self.assertTrue(h.binentries[1,2] == 1)
        h.fill((-0.5, 1.1), 2)
        self.assertTrue(h.binentries[1,2] == 2)
        self.assertTrue(h.bincontent[1,2] == 3)
        self.assertTrue(h.squaredweights[1,2] == 5)

        h.fill((0, 0))
        self.assertTrue(h.binentries[2,1] == 1)
        self.assertTrue(h.bincontent[2,1] == 1)
        self.assertTrue(h.squaredweights[2,1] == 1)


if(__name__ == "__main__"):
    unittest.main()
