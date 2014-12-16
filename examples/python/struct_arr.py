import numpy as np
import ndhist

h = ndhist.ndhist( np.array([2,2])
                 , [np.array([0,1,2], dtype=np.dtype(np.int64)),
                    np.array([0,1,2], dtype=np.dtype(np.float64))]
                 , dtype=np.dtype(np.float64))


a0 = np.array([0,1,2], dtype=np.dtype(np.int64))
a1 = np.array([0.1, 0.2, 0.3], dtype=np.dtype(np.float64))

v = np.zeros(3, dtype=h.ndvalues_dtype)
v['a0'] = a0
v['a1'] = a1

print(v)

h.fill(v, 0.5)
print(h.bc)

h.fill([(0,0.44), (1,1.7)], 1)
print(h.bc)
