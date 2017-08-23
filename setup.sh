#!/bin/bash

setupATLAS
lsetup root
source ve/bin/activate

export PATH="${PWD}:${PATH}"
export PATH="${PWD}/scripts:${PATH}"

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH}"
export PYTHONPATH="${PWD}/xgboost/python-package":${PYTHONPATH}

export THEANO_FLAGS="gcc.cxxflags='-march=core2'"
