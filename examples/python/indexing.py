import ndhist
import numpy as np

h1 = ndhist.ndhist((np.array([0,1,2], dtype=np.float64),
                    np.array([0,1,2,3], dtype=np.float64),
                    np.array([0,1,2], dtype=np.float64),
                   ), np.dtype(np.float64))
#print(h1.bincontent)
h2 = h1[(0,)]
#print(h2.bincontent)
