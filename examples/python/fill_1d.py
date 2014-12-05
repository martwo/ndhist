import numpy as np
import ndhist

# Construct a histogram with 2 bins of type int64 having edges at 0, 1, and 2,
# which are also of type int64
class Value(object):
    def __init__(self, v):
        self._v = v
    def __lt__(self, rhs):
        print("Value < %f"%(rhs._v))
        return self._v < rhs._v
    def __le__(self, rhs):
        print("Value <= %f"%(rhs._v))
        return self._v <= rhs._v

class Value2(object):
    def __init__(self, v):
        self._v = v
    def __lt__(self, rhs):
        print("Value < %f"%(rhs._v))
        return self._v < rhs._v
    def __le__(self, rhs):
        print("Value <= %f"%(rhs._v))
        return self._v <= rhs._v

h = ndhist.ndhist(np.array([2]),
                  [np.array([0,1,2], dtype=np.dtype(np.int32))],
                  #[np.array([Value(0),Value(1),Value(2)], dtype=np.dtype(object))],
                  dtype=np.dtype(np.int64))

values = np.random.uniform(0, 2, size=1000000)
vs = values.astype(np.dtype(np.int32))
vs = np.reshape(vs, (vs.shape[0],1))
#print vs.dtype
h.fill(vs, 1)
#h.fill([np.int32(1.2), np.int32(2.2)], 1)
#h.fill(np.array([[Value(-0.4)], [Value(0.4)], [Value(1.4)], [Value(2.4)]], dtype=np.dtype(object)), 1)
