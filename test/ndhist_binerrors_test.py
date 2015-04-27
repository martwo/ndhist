import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_ndhist_binerrors(self):
        """Tests of the ndhist's binerrors property works properly.

        """
        axis_0 = ndhist.axes.linear(0, 10)

        h = ndhist.ndhist((axis_0,))

        h.fill(1.2, 2)
        h.fill(2.2, 4)
        h.fill(2.3, 3)
        self.assertTrue(np.all(h.bincontent == np.array(
            [0., 2., 7., 0., 0., 0., 0., 0., 0., 0.]
        )))
        self.assertTrue(np.all(h.binerror == np.array(
            [0., 2., 5., 0., 0., 0., 0., 0., 0., 0.]
        )))

if(__name__ == "__main__"):
    unittest.main()
