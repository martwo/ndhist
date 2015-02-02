
list(APPEND ndhist_LIBRARY_LIST ndhist)
list(APPEND ndhist_LIBRARY_LIST ${BOOSTNUMPY_LIBRARIES})

add_python_module(core ${ndhist_LIBRARY_LIST}
    axis.cpp
    axes/constant_bin_width_axis.cpp
    axes/generic_axis.cpp
    error.cpp
    ndhist.cpp
    ndtable.cpp
    module.cpp
)
