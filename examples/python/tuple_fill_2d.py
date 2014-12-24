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
    def __iadd__(self, rhs):
        print("%f += %f"%(self._v, rhs._v))
        self._v += rhs._v
        return self
    def __sub__(self, rhs):
        print("%f - %f"%(self._v, rhs._v))
        return V(self._v - rhs._v)
    def __str__(self):
        return str(self._v)

h = ndhist.ndhist(((np.array([0,1,2], dtype=np.dtype(np.int64)),   "x", 1, 1),
                   (np.array([0,1,2], dtype=np.dtype(np.float64)), "y")
                  )
                  , dtype=np.dtype(object)
                  , bc_class=V)
print(h.ndvalues_dtype)
print("bc org shape:", h.bincontent.shape)

a1 = np.linspace(0, 10, num=11, endpoint=True).astype(np.dtype(np.int64))
#a1 = np.array([-2, -2, 0, 1, 2], dtype=np.dtype(np.int64))
a2 = np.array([0.1]*a1.size, dtype=np.dtype(np.float64))
print(a1, a2)
h.fill((a1, a2), V(1))

print("bc new shape", h.bincontent.shape)
bc_arr = h.bincontent
for x in range(0, bc_arr.shape[0]):
    for y in range(0, bc_arr.shape[1]):
        print("bc[%d][%d] = %f"%(x,y,bc_arr[x][y]._v))


