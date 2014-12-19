import numpy as np
import ndhist

h = ndhist.ndhist(((np.array([0,1,2], dtype=np.dtype(np.int64)),   "x", 10, 10),
                   (np.array([0,1,2], dtype=np.dtype(np.float64)), "y")
                  )
                  , dtype=np.dtype(np.float64))
print(h.ndvalues_dtype)
print(h.bc)

a1 = np.array(np.linspace(0,2, num=100), dtype=np.dtype(np.int64))
a2 = np.linspace(0,2, num=100)
print(a1, a2)
h.fill((a1, a2), 1)
print(h.bc)
