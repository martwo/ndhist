
list(APPEND ndhist_LIBRARY_LIST ndhist)
list(APPEND ndhist_LIBRARY_LIST ${BOOSTNUMPY_LIBRARIES})

add_python_module(ndhist ${ndhist_LIBRARY_LIST}
    error.cpp
    ndhist.cpp
    ndtable.cpp
    module.cpp
)
