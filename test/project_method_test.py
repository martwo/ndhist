import unittest

import numpy as np
import ndhist

class Test(unittest.TestCase):
    def test_project_method(self):
        """Tests if the project method of the ndhist class works.

        """
        axis_0 = ndhist.axes.linear(-2, 3, 1)
        axis_1 = ndhist.axes.linear(-1, 2, 1)

        h = ndhist.ndhist((axis_0, axis_1))

        # Fill into bin [1,2]
        h.fill((-1, 1))
        h.fill((-0.5, 1.1), 2)

        # Fill into bin [2,1]
        h.fill((0, 0))

        # Fill into bin [2,2]
        h.fill((0, 1.3), 3)

        p0 = h.project(0)
        self.assertTrue(p0.nbins == (5,))
        self.assertTrue(p0.bincontent[0] == 0)
        self.assertTrue(p0.bincontent[1] == 3)
        self.assertTrue(p0.bincontent[2] == 4)
        self.assertTrue(p0.bincontent[3] == 0)
        self.assertTrue(p0.bincontent[4] == 0)

        p1 = h.project(1)
        self.assertTrue(p1.nbins == (3,))
        self.assertTrue(p1.bincontent[0] == 0)
        self.assertTrue(p1.bincontent[1] == 1)
        self.assertTrue(p1.bincontent[2] == 6)

if(__name__ == "__main__"):
    unittest.main()
