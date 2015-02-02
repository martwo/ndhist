ndhist
======

A multi-dimensional histogram object usable in C++ and Python.

Dependencies
------------

ndhist depends on the following libraries:

- cmake (>= 2.8.3)
- python (>= 2.7)
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

The Python package of ndhist can be installed either relative to the specified
``--prefix`` directory, or into the user's home directory. In order to install
it into the user's python package repository, the ``--user`` option needs to
be specified.

After success, change to the build directory::

    cd build

Start the compilation process::

    make

Run all the tests::

    make test

After that, install ndhist by typing::

    make install
