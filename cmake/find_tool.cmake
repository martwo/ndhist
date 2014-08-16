#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <ndhist@martin-wolf.org>
#
# This file implements the cmake find_tool function.
# The find_tool function searches for a tool's include and library directories.
#
# This file is distributed under the BSD 2-Clause Open Source License
# (See LICENSE file).
#

#_______________________________________________________________________________
macro(print_not_found_msg msg)
    message(STATUS "- ${msg}")
endmacro()

macro(print_found_msg msg)
    message(STATUS "+ ${msg}")
endmacro()

#_______________________________________________________________________________
function(find_tool tool_ incdir_ incfile_ libdir_)
    message(STATUS "${tool_}")

    string(TOUPPER ${tool_} TOOL)

    # Set the config error to false by default
    set(${TOOL}_CONFIG_ERROR FALSE)

    # Search for the tool's include directory.
    if(NOT "${incdir_}" STREQUAL "NONE")
        find_path(${TOOL}_INCLUDE_DIR
            NAMES ${incfile_}
            PATHS ${incdir_} ENV ${TOOL}_INCLUDE_PATH
            DOC "The ${tool_} include directory.")
        if(${${TOOL}_INCLUDE_DIR} MATCHES ".*NOTFOUND$")
            print_not_found_msg("${incfile_} not found in ${incdir_}")
            set(${TOOL}_CONFIG_ERROR TRUE)
        else()
            print_found_msg("${incfile_} found at ${${TOOL}_INCLUDE_DIR}")
        endif()
    else()
        set(${TOOL}_INCLUDE_DIR "/doesnt/exist"
            CACHE PATH "The ${tool_} include directory.")
    endif()

    # Search for the tool's library directory for each library specified as
    # additional function argument.
    set(${TOOL}_LIB_LIST)
    foreach(lib ${ARGN})
        find_library(${TOOL}_foundlib${lib}
                NAMES ${lib}
                PATHS ${libdir_} ENV ${TOOL}_LIBRARY_PATH
                DOC "The ${tool_}'s ${lib} library directory.")

        if(${${TOOL}_foundlib${lib}} MATCHES ".*NOTFOUND$" AND NOT ${libdir_} STREQUAL "NONE")
            print_not_found_msg("library '${lib}'")
            set(${TOOL}_CONFIG_ERROR TRUE)
        else()
            if(NOT ${libdir_} STREQUAL "NONE")
                print_found_msg("library '${${TOOL}_foundlib${lib}}'")
                list(APPEND ${TOOL}_LIB_LIST ${${TOOL}_foundlib${lib}})
            endif()
        endif()
    endforeach()
    set(${TOOL}_LIBRARIES "${${TOOL}_LIB_LIST}"
        CACHE PATH "Libraries for tool ${TOOL}" FORCE)

    # Save the flag if the tool found found successfully.
    if(NOT ${TOOL}_CONFIG_ERROR)
        set(${TOOL}_FOUND TRUE CACHE BOOL "Tool '${tool_}' found successfully" FORCE)
    else()
        set(${TOOL}_FOUND FALSE CACHE BOOL "Tool '${tool_}' found successfully" FORCE)
    endif()

endfunction()
