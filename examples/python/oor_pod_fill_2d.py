import numpy as np
import ndhist

h = ndhist.ndhist(((np.array([0,1,2], dtype=np.dtype(np.int64)),   "x", 1, 1),
                   (np.array([0,1,2], dtype=np.dtype(np.float64)), "y")
                  )
                  , dtype=np.dtype(np.float64)
                 )
print(h.ndvalues_dtype)
print("bc org shape:", h.bincontent.shape)

a1 = np.linspace(0, 10, num=11, endpoint=True).astype(np.dtype(np.int64))
#a1 = np.array([-2, -2, 0, 1, 2], dtype=np.dtype(np.int64))
a2 = np.array([0.1]*a1.size, dtype=np.dtype(np.float64))
print(a1, a2)
h.fill((a1, a2), 1)

print("bc new shape", h.bincontent.shape)
print(h.bincontent)

h2 = ndhist.ndhist(((np.array([0,1,2], dtype=np.dtype(np.int64)),   "x", 0, 0),
                   (np.array([0,1,2], dtype=np.dtype(np.float64)), "y")
                  )
                  , dtype=np.dtype(np.float64)
                 )
h2.fill((a1, a2), 1)
print(h2.bincontent)
