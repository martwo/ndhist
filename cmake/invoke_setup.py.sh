# Invoke this script like
# invoke_setup.py.sh ${INSTALL_PYTHON_PACKAGE_INTO_USER_HOME} ${CMAKE_BINARY_DIR} ${CMAKE_INSTALL_PREFIX}

if [ "$1" == "ON" ]; then
    echo "Installing ndhist python package into the user's home directory ..."
    /usr/bin/env python $2/python/setup.py install --user
else
    # First determine the install_dir from distutils.
    OPT_INSTALL_LIB=`/usr/bin/env python $2/python/get_install_lib.py --prefix $3`
    echo "Determined install-lib directory: $OPT_INSTALL_LIB"

    mkdir -p $OPT_INSTALL_LIB
    export PYTHONPATH=$OPT_INSTALL_LIB:$PYTHONPATH
    /usr/bin/env python $2/python/setup.py install --prefix $3 --install-lib $OPT_INSTALL_LIB
fi
