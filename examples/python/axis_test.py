import numpy as np
import ndhist

a = ndhist.constant_bin_width_axis(np.array([0.,1.,2.,3.]))
print(a.nbins)
h = ndhist.ndhist((a,))
print(h.nbins)
