ndhist
======

A multi-dimensional histogram object usable in C++ and Python.

Dependencies
------------

ndhist depends on the following libraries:

- cmake (>= 2.8.3)
- python (>= 2.6)
- python-numpy (>= 1.6)
- boost (including boost::python) (>= 1.38)
- BoostNumpy (>= 0.3)

Installation
------------

An automated compilation and installation procedure is provided via cmake.
In order to create a build directory with the neccessary cmake and make files,
execute the ``configure`` script within the BoostNumpy source root directory::

    ./configure --prefix </path/to/the/installation/location>

The location of the final installation can be specified via the ``--prefix``
option. If this option is not specified, it will be installed inside the ./build
directory.

In cases where BoostNumpy is installed in special locations, one should provide
the include and library paths of the BoostNumpy installation via the configure
options ``--boostnumpy-include-path`` and ``--boostnumpy-library-path``.

After success, change to the build directory::

    cd build

Start the compilation process::

    make

Run all the tests::

    make test

After that, install ndhist by typing::

    make install
