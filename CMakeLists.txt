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

# Set NDHist_BINARY_DIR and NDHist_SOURCE_DIR cmake variables.
project(NDHist CXX)

message(STATUS "Configuring project \"${PROJECT_NAME}\"")
message(STATUS "+    project source dir: ${PROJECT_SOURCE_DIR}")

set(NDHIST_CONFIGURED_SUCCESSFULLY TRUE
    CACHE BOOL "${PROJECT_NAME} configured successfully." FORCE)

include(${CMAKE_SOURCE_DIR}/cmake/config.cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CACHE PATH "Output directory for shared libraries." FORCE)

include(${CMAKE_SOURCE_DIR}/cmake/tools/python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/boostnumpy.cmake)

find_python()
find_boostnumpy()
