import numpy as np
import ndhist

h = ndhist.ndhist((np.array([0,1,2,3,4,5,6,7,8,9,11], dtype=np.dtype(np.float64)),
                  )
                  , dtype=np.dtype(np.float64))
h.fill([-0.1, 0, 0.9, 1, 3.3, 9.9, 10, 11.1])
print(h.bc)

class V(object):
    def __init__(self, v=0):
        self._v = v

    def __lt__(self, rhs):
        print("%f < %f"%(self._v, rhs._v))
        return self._v < rhs._v

    def __add__(self, rhs):
        print("%f + %f"%(self._v, rhs._v))
        return V(self._v + rhs._v)


h2 = ndhist.ndhist((np.array([V(0),V(1),V(2),V(3),V(4),V(5),V(6),V(7),V(8),V(9),V(10)], dtype=np.dtype(object)),
                   )
                   , dtype=np.dtype(np.float64))
h2.fill([V(-0.1), V(0), V(0.9), V(1), V(3.3), V(9.9), V(10), V(11.1)] )
print(h2.bc)
