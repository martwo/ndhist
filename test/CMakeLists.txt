# Copyright (C)
# 2015 - $Date$
#     Martin Wolf <ndhist@martin-wolf.org>
#
# This file implements cmake configuration for building the unit tests of the
# ndhist tool.
#
# This file is distributed under the BSD 2-Clause Open Source License
# (See LICENSE file).
#
set(${PROJECT_NAME}_RUN_PYTHON_TEST_SCRIPT
    run_python_test.sh)
configure_file(${${PROJECT_NAME}_RUN_PYTHON_TEST_SCRIPT}.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_RUN_PYTHON_TEST_SCRIPT} @ONLY)

set(${PROJECT_NAME}_TEST_SOURCE_DIR
    ${PROJECT_SOURCE_DIR}/test)
set(${PROJECT_NAME}_PYTHON_TEST_COMMAND
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_RUN_PYTHON_TEST_SCRIPT}")

list(APPEND ${PROJECT_NAME}_PYMOD_ADD_LIB_LIST
    ndhist
    ${BOOST_LIBRARIES})

function(add_python_test _NAME _PYSCRIPT)

    # Make a test of the module using the python source file in the test
    # directory.
    message(STATUS "Adding python test: ${${PROJECT_NAME}_PYTHON_TEST_COMMAND} ${${PROJECT_NAME}_TEST_SOURCE_DIR}/${_PYSCRIPT}")
    add_test(${_NAME} ${${PROJECT_NAME}_PYTHON_TEST_COMMAND} ${${PROJECT_NAME}_TEST_SOURCE_DIR}/${_PYSCRIPT})

    # Set the regex to use to recognize a failure since `python testfoo.py`
    # does not seem to return non-zero with a test failure.
    set_property(TEST ${_NAME} PROPERTY FAIL_REGULAR_EXPRESSION "ERROR\\:")

endfunction(add_python_test)

add_python_test(nbins_test                         nbins_test.py)
add_python_test(ndhist_basic_slicing_test          ndhist_basic_slicing_test.py)
add_python_test(ndhist_clear_method_test           ndhist_clear_method_test.py)
add_python_test(ndhist_deepcopy_method_test        ndhist_deepcopy_method_test.py)
add_python_test(ndhist_merge_axis_bins_method_test ndhist_merge_axis_bins_method_test.py)
add_python_test(oor_bin_copies_test                oor_bin_copies_test.py)
add_python_test(project_method_test                project_method_test.py)
add_python_test(tuple_fill_test                    tuple_fill_test.py)
