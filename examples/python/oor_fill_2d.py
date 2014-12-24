import numpy as np
import ndhist

h = ndhist.ndhist(((np.array([0,1,2], dtype=np.dtype(np.int64)),   "x"),
                   (np.array([0,1,2], dtype=np.dtype(np.float64)), "y")
                  )
                  , dtype=np.dtype(np.float64)
                 )

a1 = np.linspace(0, 10, num=11, endpoint=True).astype(np.dtype(np.int64))
a2 = np.array([0.1]*a1.size, dtype=np.dtype(np.float64))
print(a1, a2)

h.fill((a1, a2), 1)

print(h.bincontent)
