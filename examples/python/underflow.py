import ndhist
import numpy as np

h1 = ndhist.ndhist((np.array([0,1,2,3], dtype=np.float64),
                    np.array([0,1,2], dtype=np.float64)
                   ), np.dtype(np.float64))

print(h1.nbins)
print("underflow_entries: %s"%str(h1.underflow_entries))
print("underflow: %s"%str(h1.underflow))
print("underflow_squaredweights: %s"%str(h1.underflow_squaredweights))
print("overflow_entries: %s"%str(h1.overflow_entries))
print("overflow: %s"%str(h1.overflow))
print("overflow_squaredweights: %s"%str(h1.overflow_squaredweights))
print("========= Fill Not-OOR value")
h1.fill((1,0.4), 2)
print("underflow: %s"%str(h1.underflow))
print("overflow: %s"%str(h1.overflow))

print("========= Fill X-underflow OOR value")
h1.fill((-1.2,0.2), 2)
print("underflow: %s"%str(h1.underflow))
print("overflow: %s"%str(h1.overflow))

print("========= Fill X-overflow & Y-overflow OOR value")
h1.fill((3.2,2.1), 2)
print("underflow_entries: %s"%str(h1.underflow_entries))
print("underflow: %s"%str(h1.underflow))
print("underflow_squaredweights: %s"%str(h1.underflow_squaredweights))
print("overflow_entries: %s"%str(h1.overflow_entries))
print("overflow: %s"%str(h1.overflow))
print("overflow_squaredweights: %s"%str(h1.overflow_squaredweights))
