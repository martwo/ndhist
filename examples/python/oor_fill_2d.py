import numpy as np
import ndhist
import dashi
import time
import resource

h = ndhist.ndhist(((np.array([-2,-1,0,1,2], dtype=np.dtype(np.float64)), "x", 0, 0),
                   (np.array([-2,-1,0,1,2], dtype=np.dtype(np.float64)), "y", 0, 0)
                  )
                  , dtype=np.dtype(np.float64)
                 )

d = dashi.histogram.histogram(  2
                    , (np.array([-2,-1,0,1,2], dtype=np.dtype(np.float64)),
                       np.array([-2,-1,0,1,2], dtype=np.dtype(np.float64))
                      )
                   )

a1 = np.random.normal(0, 0.5, size=1e7) #np.linspace(-10, 10, num=21, endpoint=True).astype(np.dtype(np.int64))
a2 = np.random.normal(0, 0.5, size=1e7) #np.linspace(0, 2, num=a1.size).astype(np.dtype(np.float64))

# Make sure all values are within the range.
a1 = np.select([a1 < -2, a1 >= -2], [-2, a1])
a1 = np.select([a1 >= 2, a1 < 2], [1, a1])
print(np.any(a1[a1<-2]))
print(np.any(a1[a1>=2]))

a2 = np.select([a2 < -2, a2 >= -2], [-2, a2])
a2 = np.select([a2 >= 2, a2 < 2], [1, a2])
print(np.any(a2[a2<-2]))
print(np.any(a2[a2>=2]))

print(a1, a2)

#time.sleep(3)
print("Filling h")
h_ru_start = resource.getrusage(resource.RUSAGE_SELF)
h.fill((a1, a2), 1)
h_ru_end = resource.getrusage(resource.RUSAGE_SELF)
h_dutime = h_ru_end[0]-h_ru_start[0]
h_dstime = h_ru_end[1]-h_ru_start[1]
h_total = h_dutime+h_dstime
print("done. Dutime = %g, Dstime = %g, Total = %g"%(h_dutime, h_dstime, h_total))

#time.sleep(3)
print("Filling d")
d_ru_start = resource.getrusage(resource.RUSAGE_SELF)
d.fill((a1, a2), None)
d_ru_end = resource.getrusage(resource.RUSAGE_SELF)
d_dutime = d_ru_end[0]-d_ru_start[0]
d_dstime = d_ru_end[1]-d_ru_start[1]
d_total = d_dutime+d_dstime
print("done. Dutime = %g, Dstime = %g, Total = %g"%(d_dutime, d_dstime, d_total))

print("Ratio h/d = %g"%(h_total/d_total))

#time.sleep(3)
print(h.bincontent)
print(d.bincontent)
