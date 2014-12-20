import numpy as np
import ndhist

class V(object):
    def __init__(self, v=0):
        self._v = v

    def __lt__(self, rhs):
        print("%f < %f"%(self._v, rhs._v))
        return self._v < rhs._v

    def __add__(self, rhs):
        print("%f + %f"%(self._v, rhs._v))
        return V(self._v + rhs._v)
    def __sub__(self, rhs):
        print("%f - %f"%(self._v, rhs._v))
        return V(self._v - rhs._v)

h = ndhist.ndhist(((np.array([0,1,2], dtype=np.dtype(np.int64)),   "x", 10, 10),
                   (np.array([0,1,2], dtype=np.dtype(np.float64)), "y")
                  )
                  , dtype=np.dtype(object)
                  , bc_class=V)
print(h.ndvalues_dtype)
print(h.bc)

a1 = np.array([-1, 0, 1, 2], dtype=np.dtype(np.int64))
a2 = np.array([0.1, 1.1, 1.2, 1.8], dtype=np.dtype(np.float64))
print(a1, a2)
h.fill((a1, a2), V(1))
print(h.bc)
