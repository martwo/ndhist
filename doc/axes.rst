.. _ndhist_axes:

axes
====

Each n-dimensional histogram has n axes. This axes are supposed to be (derived)
objects of the ``ndhist::axis`` class. The ``ndhist::axis`` class defines the
API of an axis. By having an extra class for an axis, different kind of axes can
implemented separately and independently, e.g. linear scale or logaritmic scale
axes.
