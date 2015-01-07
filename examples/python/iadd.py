import ndhist
import numpy as np

h1 = ndhist.ndhist((np.array([0,1,2], dtype=np.float64),), np.dtype(np.float64))
h2 = h1.empty_like()

h1.fill((1.0,))
h1.fill((1.4,))
print(h1.bincontent)
h2.fill((0.5,))
print(h2.bincontent)

h1 += h2
print(h1.bincontent)
