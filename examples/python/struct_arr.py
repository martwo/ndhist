import numpy as np
import ndhist

a = np.zeros(3, dtype=[('x',np.float64),('y',np.float32),('value',np.float16,(2,2))])

h = ndhist.ndhist(  np.array([2])
                 , [np.array([0,1,2], dtype=np.dtype(np.float64))]
                 , dtype=np.dtype(np.float64))
h.handle_struct_array(a)
