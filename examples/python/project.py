import ndhist
import numpy as np

h1 = ndhist.ndhist((np.array([0,1,2], dtype=np.float64),
                    np.array([0,1,2,3], dtype=np.float64),
                    np.array([0,1,2], dtype=np.float64)
                   ), np.dtype(np.float64))
h1.fill((0, 1, 1))
h1.fill((1, 2, 0))
h1.fill((1, 2, 1))
print("h1.bincontent %s"%str(h1.bincontent))

h2 = h1.project(0)
print("h2.bincontent %s"%str(h2.bincontent))
