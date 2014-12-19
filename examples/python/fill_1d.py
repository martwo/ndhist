import numpy as np
import ndhist
import sys

# Construct a histogram with 2 bins of type int64 having edges at 0, 1, and 2,
# which are also of type int64
class Value(object):
    def __init__(self, v=0):
        self._v = v

    def __lt__(self, rhs):
        print("%f < %f"%(self._v, rhs._v))
        return self._v < rhs._v
    def __add__(self, rhs):
        print("%f + %f"%(self._v, rhs._v))
        self._v += rhs._v
        return self

h1 = ndhist.ndhist((np.array([-1, 0,1,2,3,4,5,6,7,8,9], dtype=np.dtype(np.float64)),),
                   dtype=np.dtype(np.float64))

vs = np.random.uniform(-3, 11, size=100)
vs = vs.astype(np.dtype(np.float64))
vs = np.reshape(vs, (vs.shape[0],1))

h1.fill(vs, 1)

print("h1.get_bin_edges(0) = ", h1.get_bin_edges(0))
print("h1.bc =", h1.bc)

h2 = ndhist.ndhist((np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.dtype(np.float64)),),
                   dtype=np.dtype(np.float64))

values = np.array([1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0])
vs = values.astype(np.dtype(np.float64))
vs = np.reshape(vs, (vs.shape[0],1))
h2.fill(vs, 1)
print(h2.bc)

h3 = ndhist.ndhist((np.array([0,1,2,3,4,5,6,7,8,9,10], dtype=np.dtype(np.float64)),),
                   dtype=np.dtype(Value),
                   bc_class=Value)
print("edges h3=", h3.get_bin_edges())
print("h3.nd=", h3.nd)

values = np.array([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0])
vs = values.astype(np.dtype(np.float64))
vs = np.reshape(vs, (vs.shape[0],1))
h3.fill(vs, Value(1))
print(h3.bc)
print("[")
for i in range(0, 10):
    print("%f,"%h3.bc[i]._v)
print("]")

h4 = ndhist.ndhist((np.array([Value(0),Value(1),Value(2)], dtype=np.dtype(object)),),
                   dtype=np.dtype(np.float64))
vs = np.array([Value(0.1),Value(1.2),Value(2.3)])
vs = np.reshape(vs, (vs.shape[0],1))
h4.fill(vs, 1.0)
print("h4.bc = ", h4.bc)
