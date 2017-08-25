#!/bin/bash

setupATLAS
lsetup "root 6.04.14-x86_64-slc6-gcc49-opt"
lsetup "gcc gcc493_x86_64_slc6"
source ve/bin/activate

export PATH="${PWD}:${PATH}"
export PATH="${PWD}/scripts:${PATH}"

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH}"
export PYTHONPATH="${PWD}/xgboost/python-package":${PYTHONPATH}

export THEANO_FLAGS="gcc.cxxflags='-march=core2'"
