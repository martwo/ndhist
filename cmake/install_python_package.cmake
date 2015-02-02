message(STATUS "Installing python package \"ndhist\" via setuptools.")

execute_process(
    COMMAND python ${CMAKE_BINARY_DIR}/python/setup.py install --user
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/python
    ERROR_VARIABLE ${PROJECT}_SETUPTOOLS_ERR
    OUTPUT_VARIABLE ${PROJECT}_SETUPTOOLS_OUT
)

message(STATUS "setuptools output: ${${PROJECT}_SETUPTOOLS_OUT}")
#message(STATUS "setuptools error: ${${PROJECT}_SETUPTOOLS_ERR}")

