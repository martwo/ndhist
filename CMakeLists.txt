#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <ndhist@martin-wolf.org>
#
# This file implements cmake configuration for building the NDHist tool.
#
# This file is distributed under the BSD 2-Clause Open Source License
# (See LICENSE file).
#
cmake_minimum_required(VERSION 2.8.3)

# Choose CMAKE_BUILD_TYPE=Release if the user does not specify it.
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

# Set ndhist_BINARY_DIR and ndhist_SOURCE_DIR cmake variables.
project(ndhist CXX)

message(STATUS "Configuring project \"${PROJECT_NAME}\"")
message(STATUS "+    project source dir: ${PROJECT_SOURCE_DIR}")

option(INSTALL_PYTHON_PACKAGE_INTO_USER_HOME
       "If set to ON, the python package \"ndhist\" will be installed into the user's home directory."
       OFF)
if(INSTALL_PYTHON_PACKAGE_INTO_USER_HOME)
    message(STATUS "+    Will install the ${PROJECT_NAME} python package into the user's home directory.")
else()
    message(STATUS "+    Will install the ${PROJECT_NAME} python package within the '${CMAKE_INSTALL_PREFIX}' directory.")
endif()

option(USE_MAINTAINER_MODE "Enable warnings usually relevant to developers" OFF)
if(USE_MAINTAINER_MODE)
    message(STATUS "Enable maintainer mode.")
    set(MAINTAINER_MODE_WARN_FLAGS
        extra
        unused-parameter
        unused-variable
        missing-field-initializers
        declaration-after-statement
        missing-declarations
        redundant-decls
        cast-align
    )
    foreach(flag ${MAINTAINER_MODE_WARN_FLAGS})
        set(cxxflag "-W${flag}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxxflag}")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${cxxflag}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${cxxflag}")
        set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS} ${cxxflag}")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} ${cxxflag}")
    endforeach()
endif()

# Enable compiler optimizations.
set(cxxflag "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxxflag}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${cxxflag}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${cxxflag}")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS} ${cxxflag}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} ${cxxflag}")

set(${PROJECT_NAME}_CONFIGURED_SUCCESSFULLY TRUE)

include(${CMAKE_SOURCE_DIR}/cmake/config.cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CACHE PATH "Output directory for shared libraries." FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CACHE PATH "Output directory for executables like tests.")

include(${CMAKE_SOURCE_DIR}/cmake/add_python_module.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/find_tool.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/numpy.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/boostnumpy.cmake)

find_boost()
find_python()
find_numpy()
find_boostnumpy()

if(NOT (BOOST_FOUND AND PYTHON_FOUND AND NUMPY_FOUND AND BOOSTNUMPY_FOUND))
    set(${PROJECT_NAME}_CONFIGURED_SUCCESSFULLY FALSE)
endif()

if(${PROJECT_NAME}_CONFIGURED_SUCCESSFULLY)

    list(APPEND ${PROJECT_NAME}_libndhist_INCLUDE_DIRS
        ${BOOST_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
        ${BOOSTNUMPY_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )

    list(APPEND ${PROJECT_NAME}_libndhist_LIBRARIES
        ${PYTHON_LIBRARIES}
        ${BOOST_LIBRARIES}
        ${BOOSTNUMPY_LIBRARIES}
    )

    list(APPEND ${PROJECT_NAME}_libndhist_SOURCE_FILES
        src/ndhist/detail/bytearray.cpp
        src/ndhist/detail/ndarray_storage.cpp
        src/ndhist/ndhist.cpp
        src/ndhist/ndtable.cpp
        src/ndhist/storage.cpp
        src/ndhist/stats/excess.cpp
        src/ndhist/stats/expectation.cpp
        src/ndhist/stats/kurtosis.cpp
        src/ndhist/stats/mean.cpp
        src/ndhist/stats/median.cpp
        src/ndhist/stats/skewness.cpp
        src/ndhist/stats/std.cpp
        src/ndhist/stats/var.cpp
    )

    include_directories(
        ${${PROJECT_NAME}_libndhist_INCLUDE_DIRS}
    )
    add_library(ndhist
        ${${PROJECT_NAME}_libndhist_SOURCE_FILES}
    )
    target_link_libraries(ndhist
        ${${PROJECT_NAME}_libndhist_LIBRARIES}
    )
    get_target_property(${PROJECT_NAME}_libndhist_LIBRARY_OUTPUT_DIRECTORY
        ndhist LIBRARY_OUTPUT_DIRECTORY)
    add_custom_command(TARGET ndhist
        PRE_LINK
        COMMAND mkdir -p ${${PROJECT_NAME}_libndhist_LIBRARY_OUTPUT_DIRECTORY}
    )

    # Copy some cmake files over to the build directory.
    add_custom_command(TARGET ndhist
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/cmake
    )
    add_custom_command(TARGET ndhist
        POST_BUILD
        COMMAND cp ${PROJECT_SOURCE_DIR}/cmake/invoke_setup.py.sh ${CMAKE_BINARY_DIR}/cmake
    )

    add_custom_target(build_python_package
        ALL
        mkdir -p ${CMAKE_BINARY_DIR}/python
        COMMAND cp -r ${PROJECT_SOURCE_DIR}/python ${CMAKE_BINARY_DIR}/
        DEPENDS ndhist
        COMMENT "Building python package ..."
    )

    # Install all the public header files into the include directory.
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ndhist
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".git" EXCLUDE
    )

    # Install the shared libraries into the library directory.
    install(TARGETS ndhist
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )

    # Build the core python extension module that provides the Python
    # interface to the C++ code of ndhist.
    list(APPEND ${PROJECT}_core_LIBRARY_LIST ndhist)
    list(APPEND ${PROJECT}_core_LIBRARY_LIST ${BOOSTNUMPY_LIBRARIES})
    add_python_module(core ${${PROJECT}_core_LIBRARY_LIST}
        src/pybindings/axis.cpp
        src/pybindings/axes/generic_axis.cpp
        src/pybindings/axes/linear_axis.cpp
        src/pybindings/axes/log10_axis.cpp
        src/pybindings/stats/excess.cpp
        src/pybindings/stats/expectation.cpp
        src/pybindings/stats/kurtosis.cpp
        src/pybindings/stats/mean.cpp
        src/pybindings/stats/median.cpp
        src/pybindings/stats/skewness.cpp
        src/pybindings/stats/std.cpp
        src/pybindings/stats/var.cpp
        src/pybindings/stats/module.cpp
        src/pybindings/error.cpp
        src/pybindings/ndhist.cpp
        src/pybindings/ndtable.cpp
        src/pybindings/module.cpp
    )

    # Install the python module "ndhist" via setuptools.
    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/install_python_package.cmake.in
        ${CMAKE_BINARY_DIR}/cmake/install_python_package.cmake
        @ONLY
    )
    install(SCRIPT ${CMAKE_BINARY_DIR}/cmake/install_python_package.cmake)

    enable_testing()
    add_subdirectory(test)

    message(STATUS "Project \"${PROJECT_NAME}\" configured.")
    message(STATUS "+    Now, type-in \"cd ${CMAKE_BINARY_DIR} && make && make install\"!")
    message(STATUS "+    \"${PROJECT_NAME}\" will be installed in '${CMAKE_INSTALL_PREFIX}'.")

endif()
