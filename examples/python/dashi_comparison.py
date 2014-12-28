import numpy as np
import ndhist
import dashi
import resource
import pylab

N = 1e7

axis = np.linspace(-100, 100, num=201, endpoint=True).astype(np.dtype(np.float64))
axis_min = axis.min()
axis_max = axis.max()

print("-----------------------------------------------------------")
print("Fill performance comparison of only not-out-of-range values")

h = ndhist.ndhist(( (axis, "x", "The x-axis.", 0, 0),
                   #(axis, "y", 0, 0)
                  )
                  , dtype=np.dtype(np.float64)
                 )

d = dashi.histogram.hist1d(axis)
print(h.nbins)
print(d.nbins)

a1 = np.random.normal(0, 50, size=N)
a2 = np.random.normal(0, 50, size=N)

# Make sure all values are within the range.
a1 = np.select([a1 < axis_min, a1 >= axis_min], [axis_min, a1])
a1 = np.select([a1 >= axis_max, a1 < axis_max], [axis_max-1, a1])
print("%s == False ?"%(np.any(a1[a1<axis_min])))
print("%s == False ?"%(np.any(a1[a1>=axis_max])))

a2 = np.select([a2 < axis_min, a2 >= axis_min], [axis_min, a2])
a2 = np.select([a2 >= axis_max, a2 < axis_max], [axis_max-1, a2])
print("%s == False ?"%(np.any(a2[a2<axis_min])))
print("%s == False ?"%(np.any(a2[a2>=axis_max])))



print("Filling h")
h_ru_start = resource.getrusage(resource.RUSAGE_SELF)
h.fill((a1, ), 1)
h_ru_end = resource.getrusage(resource.RUSAGE_SELF)
h_dutime = h_ru_end[0]-h_ru_start[0]
h_dstime = h_ru_end[1]-h_ru_start[1]
h_total = h_dutime+h_dstime
print("done. Dutime = %g, Dstime = %g, Total = %g"%(h_dutime, h_dstime, h_total))


print("Filling d")
d_weights = np.zeros((a1.size,), dtype=axis.dtype)+1
d_ru_start = resource.getrusage(resource.RUSAGE_SELF)
d.fill((a1, ), d_weights)
d_ru_end = resource.getrusage(resource.RUSAGE_SELF)
d_dutime = d_ru_end[0]-d_ru_start[0]
d_dstime = d_ru_end[1]-d_ru_start[1]
d_total = d_dutime+d_dstime
print("done. Dutime = %g, Dstime = %g, Total = %g"%(d_dutime, d_dstime, d_total))

print("Ratio h/d = %g"%(h_total/d_total))

print(h.bincontent)
print(d.bincontent)
print("-------------------------------------------------------")
print(h._h_bincontent)
print(d._h_bincontent)

from dashi import histviews
fig = pylab.figure()
setattr(h, "binerror", property(lambda self : np.sqrt(self._h_squaredweights[1:-1])))
histviews.h1line(h, color='red')
#histviews.h1line(d, color='blue')
pylab.show()

print("-------------------------------------------------------")
print("Fill performance comparison of only out-of-range values")

h = ndhist.ndhist(((axis, "x", "The x-axis.", 0, 0),
                   (axis, "y", "The y-axis.", 0, 0)
                  )
                  , dtype=np.dtype(np.float64)
                 )

d = dashi.histogram.histogram(  2
                    , (axis,
                       axis
                      )
                   )

a1 = np.random.normal(-200, 10, size=N)
a2 = np.random.normal(+200, 10, size=N)

# Make sure all values are outside the range.
a1 = np.select([a1 < axis_min, a1 >= axis_min], [a1, axis_min-1])
a1 = np.select([a1 >= axis_max, a1 < axis_max], [a1, axis_max])
print("%s == False ?"%(np.any(a1[(a1>=axis_min) & (a1<axis_max)])))

a2 = np.select([a2 < axis_min, a2 >= axis_min], [a2, axis_min-1])
a2 = np.select([a2 >= axis_max, a2 < axis_max], [a2, axis_max])
print("%s == False ?"%(np.any(a2[(a2>=axis_min) & (a2<axis_max)])))

print("Filling h")
h_ru_start = resource.getrusage(resource.RUSAGE_SELF)
h.fill((a1, a2), 1)
h_ru_end = resource.getrusage(resource.RUSAGE_SELF)
h_dutime = h_ru_end[0]-h_ru_start[0]
h_dstime = h_ru_end[1]-h_ru_start[1]
h_total = h_dutime+h_dstime
print("done. Dutime = %g, Dstime = %g, Total = %g"%(h_dutime, h_dstime, h_total))


print("Filling d")
d_weights = np.zeros((a1.size,), dtype=axis.dtype)+1
d_ru_start = resource.getrusage(resource.RUSAGE_SELF)
d.fill((a1, a2), d_weights)
d_ru_end = resource.getrusage(resource.RUSAGE_SELF)
d_dutime = d_ru_end[0]-d_ru_start[0]
d_dstime = d_ru_end[1]-d_ru_start[1]
d_total = d_dutime+d_dstime
print("done. Dutime = %g, Dstime = %g, Total = %g"%(d_dutime, d_dstime, d_total))

print("Ratio h/d = %g"%(h_total/d_total))

print(h.bincontent)
print(d.bincontent)
