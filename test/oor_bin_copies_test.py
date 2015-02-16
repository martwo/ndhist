import unittest

import numpy as np
import ndhist

class V(object):
    def __init__(self, v=0):
        self._v = v
    def __lt__(self, rhs):
        return self._v < rhs._v
    def __add__(self, rhs):
        return V(self._v + rhs._v)
    def __mul__(self, rhs):
        return V(self._v * rhs._v)
    def __iadd__(self, rhs):
        self._v += rhs._v
        return self
    def __sub__(self, rhs):
        return V(self._v - rhs._v)
    def __str__(self):
        return str(self._v)

class Test(unittest.TestCase):
    def test_oor_bin_copies(self):
        """Tests if the oor properties perform a real deep copy of the bins,
        when the weight type is an object, i.e. not a POD type.

        """
        axis_0 = ndhist.axes.linear(0, 10, 1)
        self.assertTrue(axis_0.nbins == 12)

        h = ndhist.ndhist((axis_0,), dtype=np.object, bc_class=V)

        u_arr_tuple1 = h.underflow
        u_arr_tuple2 = h.underflow
        self.assertTrue(len(u_arr_tuple1) == 1)
        self.assertTrue(len(u_arr_tuple2) == 1)
        u_arr1 = u_arr_tuple1[0]
        u_arr2 = u_arr_tuple2[0]
        self.assertTrue(len(u_arr1) == 1)
        self.assertTrue(len(u_arr2) == 1)
        obj1 = u_arr1[0]
        obj2 = u_arr2[0]
        self.assertTrue(isinstance(obj1, V))
        self.assertTrue(isinstance(obj2, V))
        self.assertTrue(obj1._v == 0)
        self.assertTrue(obj2._v == 0)
        obj1._v = 42
        self.assertTrue(obj1._v != obj2._v)

if(__name__ == "__main__"):
    unittest.main()
