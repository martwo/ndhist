import numpy as np
import ndhist

# Construct a histogram with 2 bins of type int64 having edges at 0, 1, and 2,
# which are also of type int64
h = ndhist.ndhist(np.array([2]),
                  [np.array([0,1,2], dtype=np.dtype(np.int64))],
                  dtype=np.dtype(np.int64))

values = np.random.uniform(0, 2, size=1000000)
values = np.reshape(values, (values.shape[0],1))
h.fill(values, 1)
