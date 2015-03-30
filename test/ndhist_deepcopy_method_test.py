import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_ndhist_deepcopy_method(self):
        """Tests if the ndhist deepcopy method works properly.

        """
        axis_0 = ndhist.axes.linear(-1, 3, 1)

        h = ndhist.ndhist((axis_0,))
        self.assertFalse(h.is_view)
        self.assertTrue(h.nbins == (4,))

        # Fill into bin [1]
        h.fill((0.6,))
        self.assertTrue(h.bincontent[0] == 0)
        self.assertTrue(h.bincontent[1] == 1)
        self.assertTrue(h.bincontent[2] == 0)
        self.assertTrue(h.bincontent[3] == 0)

        h2 = h[0:3]
        self.assertTrue(h2.is_view)
        self.assertTrue(h2.nbins == (2,))
        self.assertTrue(h2.bincontent[0] == 0)
        self.assertTrue(h2.bincontent[1] == 1)

        h3 = h2.deepcopy()
        self.assertFalse(h3.is_view)
        self.assertTrue(h3.nbins == (2,))
        self.assertTrue(h3.bincontent[0] == 0)
        self.assertTrue(h3.bincontent[1] == 1)

        h3.fill((-0.3,))
        # h2 should not have changed.
        self.assertTrue(h2.bincontent[0] == 0)
        self.assertTrue(h2.bincontent[1] == 1)
        # But h3 should have changed.
        self.assertTrue(h3.bincontent[0] == 1)
        self.assertTrue(h3.bincontent[1] == 1)

if(__name__ == "__main__"):
    unittest.main()
