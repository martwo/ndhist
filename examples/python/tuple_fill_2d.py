import numpy as np
import ndhist

h = ndhist.ndhist( np.array([2,2])
                 , [np.array([0,1,2], dtype=np.dtype(np.int64)),
                    np.array([0,1,2], dtype=np.dtype(np.float64))]
                 , dtype=np.dtype(np.float64))
#h.fill((1, 1.2), 1)
print(h.bc)

a1 = np.array(np.linspace(0,2, num=100), dtype=np.dtype(np.int64))
a2 = np.linspace(0,2, num=100)
print(a1, a2)
h.fill((a1, a2), 1)
print(h.bc)
