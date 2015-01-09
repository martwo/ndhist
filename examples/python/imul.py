import ndhist
import numpy as np

h1 = ndhist.ndhist((np.array([0,1,2], dtype=np.float64),), np.dtype(np.float64))

h1.fill((1.0,))
h1.fill((1.4,))
print(h1.bincontent)

h1 *= 3
print(h1.bincontent)
