from setuptools import setup, find_packages

setup(
    name = "ndhist",
    version = "0.1.0",
    packages = find_packages(),
    package_data = {
        # If any package contains *.so files, include them.
        '': ['*.so']
    },
    # Unpack the egg file before importing modules. This is required in order
    # to be able to import the extension modules (shared object files).
    zip_safe = False,

    # Metadata for upload to PyPI.
    author = "Martin Wolf",
    author_email = "ndhist@martin-wolf.org",
    description = "",
    license = "BSD 2-Clause",
    keywords = "histogram multi-dimensional science data-mining",
    url = "https://github.com/martwo/ndhist",
    long_description = "The Python ndhist module provides a Python interface "+\
                       "to the ndhist tool, which is written in C++. ndhist "+\
                       "provides a n-dimensional histogram class with numpy "+\
                       "array interface support.",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD 2-Clause",
        "Programming Language :: Python :: 2.7"
    ]
)
