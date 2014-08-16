#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <ndhist@martin-wolf.org>
#
# This file implements the cmake find_boostnumpy function searching for
# the BoostNumpy tool using the find_tool function.
#
# This file is distributed under the BSD 2-Clause Open Source License
# (See LICENSE file).
#
function(find_boostnumpy)

    find_tool(boostnumpy
        include
        boost/numpy.hpp
        lib
        boost_numpy
    )

endfunction(find_boostnumpy)
