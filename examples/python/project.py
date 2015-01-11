import ndhist
import numpy as np

h1 = ndhist.ndhist((np.array([0,1,2], dtype=np.float64),
                    np.array([0,1,2,3], dtype=np.float64),
                    np.array([0,1,2], dtype=np.float64),
                   ), np.dtype(np.float64))
h1.fill((0, 1, 1))
h1.fill((1, 2, 0))
h1.fill((1, 2, 1))
# Fill OOR values.
h1.fill((-1, 2, 1))
h1.fill((1, -2, 1))
h1.fill((1, 2, -1))
print("h1.bincontent %s"%str(h1.bincontent))
print("h1.underflow %s"%str(h1.underflow))
print("h1.overflow %s"%str(h1.overflow))

h2 = h1.project((0))
print("h2.bincontent %s"%str(h2.bincontent))
print("h2.underflow %s"%str(h2.underflow))
print("h2.overflow %s"%str(h2.overflow))
