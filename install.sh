setupATLAS
lsetup root
lsetup python
virtualenv --python=python2.7 ve
source ve/bin/activate

pip install pip --upgrade
pip install theano keras h5py sklearn matplotlib tabulate
pip install --upgrade https://github.com/rootpy/root_numpy/zipball/master
