import numpy as np
import ndhist
import dashi
import time
import resource

axis = np.linspace(-100, 100, num=201, endpoint=True).astype(np.dtype(np.float64))
print(axis)

h = ndhist.ndhist(((axis, "x", 0, 0),
                   (axis, "y", 0, 0)
                  )
                  , dtype=np.dtype(np.float64)
                 )

d = dashi.histogram.histogram(  2
                    , (axis,
                       axis
                      )
                   )

a1 = np.random.normal(0, 50, size=1e7) #np.linspace(-10, 10, num=21, endpoint=True).astype(np.dtype(np.int64))
a2 = np.random.normal(0, 50, size=1e7) #np.linspace(0, 2, num=a1.size).astype(np.dtype(np.float64))

# Make sure all values are within the range.
axis_min = axis.min()
axis_max = axis.max()

a1 = np.select([a1 < axis_min, a1 >= axis_min], [axis_min, a1])
a1 = np.select([a1 >= axis_max, a1 < axis_max], [axis_max-1, a1])
print(np.any(a1[a1<axis_min]))
print(np.any(a1[a1>=axis_max]))

a2 = np.select([a2 < axis_min, a2 >= axis_min], [axis_min, a2])
a2 = np.select([a2 >= axis_max, a2 < axis_max], [axis_max-1, a2])
print(np.any(a2[a2<axis_min]))
print(np.any(a2[a2>=axis_max]))

print(a1, a2)

#time.sleep(3)
print("Filling h")
h_ru_start = resource.getrusage(resource.RUSAGE_SELF)
h.fill((a1, a2), 2)
h_ru_end = resource.getrusage(resource.RUSAGE_SELF)
h_dutime = h_ru_end[0]-h_ru_start[0]
h_dstime = h_ru_end[1]-h_ru_start[1]
h_total = h_dutime+h_dstime
print("done. Dutime = %g, Dstime = %g, Total = %g"%(h_dutime, h_dstime, h_total))

#time.sleep(3)
print("Filling d")
d_weights = np.zeros((a1.size,), dtype=axis.dtype)+2
d_ru_start = resource.getrusage(resource.RUSAGE_SELF)
d.fill((a1, a2), d_weights)
d_ru_end = resource.getrusage(resource.RUSAGE_SELF)
d_dutime = d_ru_end[0]-d_ru_start[0]
d_dstime = d_ru_end[1]-d_ru_start[1]
d_total = d_dutime+d_dstime
print("done. Dutime = %g, Dstime = %g, Total = %g"%(d_dutime, d_dstime, d_total))

print("Ratio h/d = %g"%(h_total/d_total))

#time.sleep(3)
print(h.bincontent)
print(d.bincontent)

print("---------------------")
print(h.binentries)
print(h.squaredweights)
